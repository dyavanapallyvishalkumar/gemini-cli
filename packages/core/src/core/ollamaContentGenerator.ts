/* eslint-disable */
import type {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
} from '@google/genai';
import { Ollama } from 'ollama';
import type { Message } from 'ollama';
import type { ContentGenerator } from './contentGenerator.js';
import type { LlmRole } from '../telemetry/llmRole.js';

/**
 * Returns true if the error indicates the model does not support tool calling.
 */
function isToolsNotSupportedError(error: unknown): boolean {
  const msg = error instanceof Error ? error.message : String(error);
  return msg.includes('does not support tools');
}

/**
 * A ContentGenerator implementation that interacts with a local Ollama server.
 */
export class OllamaContentGenerator implements ContentGenerator {
  private ollama: Ollama;
  /** Tracks whether the model supports tool calling. Starts true, flipped on first rejection. */
  private toolsSupported = true;

  constructor(private readonly model: string) {
    // Defaults to http://127.0.0.1:11434
    this.ollama = new Ollama();
    // remove the "ollama/" prefix from the resolved model if it exists
    this.model = this.model.startsWith('ollama/')
      ? this.model.replace('ollama/', '')
      : this.model;
  }

  private mapGeminiParamsToOllama(request: GenerateContentParameters): {
    messages: Message[];
    tools?: any[];
    options: any;
  } {
    const messages: Message[] = [];

    // System instruction mapping
    if (request.config?.systemInstruction) {
      let text = '';
      if (typeof request.config.systemInstruction === 'string') {
        text = request.config.systemInstruction;
      } else if ((request.config.systemInstruction as any).parts) {
        text = (request.config.systemInstruction as any).parts
          .map((p: any) => p.text || '')
          .join('\\n');
      }
      if (text) {
        messages.push({ role: 'system', content: text });
      }
    }

    // Tools mapping — only if the model supports tools
    let ollamaTools: any[] | undefined = undefined;
    if (
      this.toolsSupported &&
      request.config?.tools &&
      request.config.tools.length > 0
    ) {
      ollamaTools = [];
      for (const t of request.config.tools) {
        if ((t as any).functionDeclarations) {
          for (const fd of (t as any).functionDeclarations) {
            ollamaTools.push({
              type: 'function',
              function: {
                name: fd.name,
                description: fd.description || '',
                parameters: fd.parameters || { type: 'object', properties: {} },
              },
            });
          }
        }
      }
    }

    // Contents mapping — filter out tool-related messages when tools are not supported
    if (request.contents) {
      const contentsArr = Array.isArray(request.contents)
        ? request.contents
        : [request.contents];
      for (const content of contentsArr) {
        const c = content as any;
        const ollamaRole = c.role === 'model' ? 'assistant' : 'user';
        let text = '';
        const tool_calls: any[] = [];
        let isToolResponse = false;

        // Handle function responses (which Gemini SDK represents as user messages with functionResponse parts)
        const functionResponseParts =
          c.parts?.filter((p: any) => p.functionResponse) || [];
        if (functionResponseParts.length > 0) {
          if (!this.toolsSupported) {
            // Skip tool response messages entirely when tools are not supported
            continue;
          }
          isToolResponse = true;
          for (const part of functionResponseParts) {
            messages.push({
              role: 'tool',
              content: JSON.stringify(part.functionResponse?.response || {}),
            });
          }
        }

        const textParts = c.parts?.filter((p: any) => p.text) || [];
        const functionCallParts =
          c.parts?.filter((p: any) => p.functionCall) || [];

        if (
          !isToolResponse ||
          textParts.length > 0 ||
          functionCallParts.length > 0
        ) {
          text = textParts.map((p: any) => p.text).join('');

          // Only include function call parts if tools are supported
          if (this.toolsSupported) {
            for (const part of functionCallParts) {
              tool_calls.push({
                function: {
                  name: part.functionCall?.name,
                  arguments: part.functionCall?.args || {},
                },
              });
            }
          }

          // Don't push empty messages
          if (text || tool_calls.length > 0) {
            messages.push({
              role: ollamaRole,
              content: text,
              ...(tool_calls.length > 0 ? { tool_calls } : {}),
            });
          }
        }
      }
    }

    const options: any = {};
    if (request.config?.temperature !== undefined) {
      options.temperature = request.config.temperature;
    }
    // Omit other options for broader compatibility unless specified

    return {
      messages,
      tools: ollamaTools?.length ? ollamaTools : undefined,
      options,
    };
  }

  async generateContent(
    request: GenerateContentParameters,
    userPromptId: string,
    role: LlmRole,
  ): Promise<GenerateContentResponse> {
    const { messages, tools, options } = this.mapGeminiParamsToOllama(request);

    try {
      let response;
      try {
        response = await this.ollama.chat({
          model: this.model,
          messages,
          tools,
          options,
          stream: false,
        });
      } catch (error) {
        if (isToolsNotSupportedError(error) && this.toolsSupported) {
          // Model doesn't support tools — remember and retry without them
          this.toolsSupported = false;
          response = await this.ollama.chat({
            model: this.model,
            messages,
            options,
            stream: false,
          });
        } else {
          throw error;
        }
      }

      const parts: any[] = [];
      if (response.message?.content) {
        parts.push({ text: response.message.content });
      }
      if (
        response.message?.tool_calls &&
        response.message.tool_calls.length > 0
      ) {
        for (const toolCall of response.message.tool_calls) {
          parts.push({
            functionCall: {
              name: toolCall.function.name,
              args: toolCall.function.arguments as Record<string, any>,
            },
          });
        }
      }

      return {
        candidates: [
          {
            content: {
              role: 'model',
              parts,
            },
          },
        ],
        usageMetadata: {
          promptTokenCount: response.prompt_eval_count || 0,
          candidatesTokenCount: response.eval_count || 0,
          totalTokenCount:
            (response.prompt_eval_count || 0) + (response.eval_count || 0),
        },
      } as any;
    } catch (error) {
      throw new Error(
        `Ollama generateContent failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  async generateContentStream(
    request: GenerateContentParameters,
    userPromptId: string,
    role: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    let { messages, tools, options } = this.mapGeminiParamsToOllama(request);
    const model = this.model;
    const ollama = this.ollama;
    const self = this;

    return (async function* () {
      try {
        let responseStream;
        try {
          responseStream = await ollama.chat({
            model,
            messages,
            tools,
            options,
            stream: true,
          });
        } catch (error) {
          if (isToolsNotSupportedError(error) && self.toolsSupported) {
            // Model doesn't support tools — remember and retry without them
            self.toolsSupported = false;
            // Re-map without tools now that the flag is off
            const remapped = self.mapGeminiParamsToOllama(request);
            messages = remapped.messages;
            tools = undefined;
            options = remapped.options;
            responseStream = await ollama.chat({
              model,
              messages,
              options,
              stream: true,
            });
          } else {
            throw error;
          }
        }

        let promptTokenCount = 0;
        let candidatesTokenCount = 0;
        let totalTokenCount = 0;

        // Aggregate tool calls because Ollama streaming might chunk them,
        // but local tools are often returned atomically at the end.
        for await (const chunk of responseStream) {
          const parts: any[] = [];
          if (chunk.message?.content) {
            parts.push({ text: chunk.message.content });
          }
          if (
            chunk.message?.tool_calls &&
            chunk.message.tool_calls.length > 0
          ) {
            for (const toolCall of chunk.message.tool_calls) {
              parts.push({
                functionCall: {
                  name: toolCall.function.name,
                  args: toolCall.function.arguments as Record<string, any>,
                },
              });
            }
          }

          if (chunk.prompt_eval_count)
            promptTokenCount = chunk.prompt_eval_count;
          if (chunk.eval_count) candidatesTokenCount = chunk.eval_count;
          if (chunk.prompt_eval_count || chunk.eval_count) {
            totalTokenCount = promptTokenCount + candidatesTokenCount;
          }

          // Build functionCalls array matching the GenAI SDK getter shape
          const chunkFunctionCalls =
            chunk.message?.tool_calls?.map((tc: any) => ({
              name: tc.function.name,
              args: tc.function.arguments,
            })) ?? undefined;

          yield {
            candidates: [
              {
                content: {
                  role: 'model',
                  parts,
                },
                // Mark the last chunk with STOP so the turn processor knows we're done
                ...(chunk.done ? { finishReason: 'STOP' } : {}),
              },
            ],
            usageMetadata: {
              promptTokenCount,
              candidatesTokenCount,
              totalTokenCount,
            },
            // Expose as plain properties (not methods) to match GenAI SDK getter shape
            text: chunk.message?.content || undefined,
            functionCalls: chunkFunctionCalls,
          } as any;
        }
      } catch (error) {
        throw new Error(
          `Ollama generateContentStream failed: ${error instanceof Error ? error.message : String(error)}`,
        );
      }
    })();
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // Ollama does not have a native token counting endpoint yet.
    // We return a rough estimate (1 token per 4 chars for english).
    let totalText = '';
    if (request.contents) {
      const contentsArr = Array.isArray(request.contents)
        ? request.contents
        : [request.contents];
      for (const c of contentsArr) {
        if ((c as any).parts) {
          totalText += ((c as any).parts as any[])
            .map((p: any) => p.text || JSON.stringify(p.functionCall || {}))
            .join(' ');
        }
      }
    }
    return { totalTokens: Math.ceil(totalText.length / 4) } as any;
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    let input = '';
    if (typeof (request as any).contents === 'string') {
      input = (request as any).contents;
    } else if ((request as any).contents?.parts) {
      input = (request as any).contents.parts
        .map((p: any) => p.text || '')
        .join(' ');
    } else if (Array.isArray((request as any).contents)) {
      input = (request as any).contents
        .map((c: any) =>
          c.parts ? c.parts.map((p: any) => p.text || '').join(' ') : '',
        )
        .join(' ');
    }

    if (!input) throw new Error('No text content for embeddings');

    const response = await this.ollama.embeddings({
      model: this.model,
      prompt: input,
    });

    return {
      embeddings: [
        {
          values: response.embedding,
        },
      ],
    } as any;
  }
}
