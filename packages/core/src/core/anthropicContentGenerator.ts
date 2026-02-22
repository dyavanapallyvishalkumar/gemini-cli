/* eslint-disable */
import type {
  CountTokensResponse,
  GenerateContentResponse,
  GenerateContentParameters,
  CountTokensParameters,
  EmbedContentResponse,
  EmbedContentParameters,
} from '@google/genai';
import Anthropic from '@anthropic-ai/sdk';
import type { ContentGenerator } from './contentGenerator.js';
import type { LlmRole } from '../telemetry/llmRole.js';

import { loadAnthropicApiKey } from './apiKeyCredentialStorage.js';

export class AnthropicContentGenerator implements ContentGenerator {
  private anthropic: Anthropic | undefined;

  constructor(private readonly model: string) {
    this.model = this.model.startsWith('anthropic/')
      ? this.model.replace('anthropic/', '')
      : this.model;
  }

  private async getAnthropic(): Promise<Anthropic> {
    if (this.anthropic) return this.anthropic;
    const key =
      process.env['ANTHROPIC_API_KEY'] || (await loadAnthropicApiKey());
    if (!key) {
      throw new Error(
        "Anthropic API key is missing. Please set ANTHROPIC_API_KEY or run 'auth' to set it.",
      );
    }
    this.anthropic = new Anthropic({ apiKey: key });
    return this.anthropic;
  }

  private mapGeminiParamsToAnthropic(request: GenerateContentParameters): {
    system?: string;
    messages: Anthropic.MessageParam[];
    tools?: Anthropic.Tool[];
    temperature?: number;
    max_tokens: number;
  } {
    let systemText = '';
    const messages: Anthropic.MessageParam[] = [];

    // System instruction mapping
    if (request.config?.systemInstruction) {
      if (typeof request.config.systemInstruction === 'string') {
        systemText = request.config.systemInstruction;
      } else if ((request.config.systemInstruction as any).parts) {
        systemText = (request.config.systemInstruction as any).parts
          .map((p: any) => p.text || '')
          .join('\\n');
      }
    }

    // Tools mapping
    let anthropicTools: Anthropic.Tool[] | undefined = undefined;
    if (request.config?.tools && request.config.tools.length > 0) {
      anthropicTools = [];
      for (const t of request.config.tools) {
        if ((t as any).functionDeclarations) {
          for (const fd of (t as any).functionDeclarations) {
            anthropicTools.push({
              name: fd.name,
              description: fd.description || '',
              input_schema: fd.parameters || { type: 'object', properties: {} },
            });
          }
        }
      }
    }

    // Contents mapping
    if (request.contents) {
      const contentsArr = Array.isArray(request.contents)
        ? request.contents
        : [request.contents];
      for (const content of contentsArr) {
        const role = (content as any).role === 'model' ? 'assistant' : 'user';
        const contentBlocks: Anthropic.ContentBlockParam[] = [];

        // Handle function responses (Gemini SDK has them as user messages with functionResponse parts)
        const functionResponseParts =
          (content as any).parts?.filter((p: any) => p.functionResponse) || [];
        if (functionResponseParts.length > 0) {
          // In anthropic, tool responses are user messages containing tool_result blocks
          for (const part of functionResponseParts) {
            contentBlocks.push({
              type: 'tool_result',
              tool_use_id: part.functionResponse.id || 'unknown_id', // Anthropic requires an ID
              content: JSON.stringify(part.functionResponse.response || {}),
            });
          }
        }

        const textParts =
          (content as any).parts?.filter((p: any) => p.text) || [];
        const functionCallParts =
          (content as any).parts?.filter((p: any) => p.functionCall) || [];

        if (textParts.length > 0) {
          contentBlocks.push({
            type: 'text',
            text: textParts.map((p: any) => p.text).join(''),
          });
        }

        for (const part of functionCallParts) {
          contentBlocks.push({
            type: 'tool_use',
            id: part.functionCall.id || 'unknown_id', // Needs to map to an ID, generative models aren't strict on user side
            name: part.functionCall.name,
            input: part.functionCall.args || {},
          });
        }

        if (contentBlocks.length > 0) {
          messages.push({
            role,
            content: contentBlocks,
          });
        }
      }
    }

    return {
      system: systemText || undefined,
      messages,
      tools: anthropicTools?.length ? anthropicTools : undefined,
      temperature: request.config?.temperature,
      max_tokens: 8192, // Ensure a high enough limit for code agent tasks
    };
  }

  async generateContent(
    request: GenerateContentParameters,
    userPromptId: string,
    role: LlmRole,
  ): Promise<GenerateContentResponse> {
    const params = this.mapGeminiParamsToAnthropic(request);

    try {
      const client = await this.getAnthropic();
      const response = await client.messages.create({
        model: this.model,
        system: params.system,
        messages: params.messages,
        tools: params.tools,
        temperature: params.temperature,
        max_tokens: params.max_tokens,
      });

      const parts: any[] = [];
      for (const block of response.content) {
        if (block.type === 'text') {
          parts.push({ text: block.text });
        } else if (block.type === 'tool_use') {
          parts.push({
            functionCall: {
              name: block.name,
              args: block.input as Record<string, any>,
              id: block.id,
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
          promptTokenCount: response.usage.input_tokens,
          candidatesTokenCount: response.usage.output_tokens,
          totalTokenCount:
            response.usage.input_tokens + response.usage.output_tokens,
        },
      } as any;
    } catch (error) {
      throw new Error(
        `Anthropic generateContent failed: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  async generateContentStream(
    request: GenerateContentParameters,
    userPromptId: string,
    role: LlmRole,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    const params = this.mapGeminiParamsToAnthropic(request);
    const model = this.model;
    const anthropic = await this.getAnthropic();

    return (async function* () {
      try {
        const stream = await anthropic.messages.stream({
          model,
          system: params.system,
          messages: params.messages,
          tools: params.tools,
          temperature: params.temperature,
          max_tokens: params.max_tokens,
        });

        let promptTokenCount = 0;
        let candidatesTokenCount = 0;
        let totalTokenCount = 0;

        let currentText = '';
        let currentToolCalls: Array<{
          name: string;
          args: Record<string, any>;
          id: string;
        }> = [];

        for await (const chunk of stream) {
          if (chunk.type === 'message_start' && chunk.message.usage) {
            promptTokenCount = chunk.message.usage.input_tokens;
          } else if (
            chunk.type === 'content_block_delta' &&
            chunk.delta.type === 'text_delta'
          ) {
            currentText += chunk.delta.text;
          } else if (chunk.type === 'message_delta' && chunk.usage) {
            candidatesTokenCount = chunk.usage.output_tokens;
          }
          // We let the complete tool calls finish from the final message event to simplify
          totalTokenCount = promptTokenCount + candidatesTokenCount;

          // Yield partial streams
          yield {
            candidates: [
              {
                content: {
                  role: 'model',
                  parts: [{ text: currentText }],
                },
              },
            ],
            usageMetadata: {
              promptTokenCount,
              candidatesTokenCount,
              totalTokenCount,
            },
            text: () => currentText,
            functionCalls: () => currentToolCalls,
          } as any;
        }

        // Final yield with complete tools
        const finalMessage = await stream.finalMessage();
        if (finalMessage.usage) {
          candidatesTokenCount = finalMessage.usage.output_tokens;
          totalTokenCount = promptTokenCount + candidatesTokenCount;
        }

        const parts: any[] = [];
        for (const block of finalMessage.content) {
          if (block.type === 'text') {
            parts.push({ text: block.text });
          } else if (block.type === 'tool_use') {
            currentToolCalls.push({
              name: block.name,
              args: block.input as Record<string, any>,
              id: block.id,
            });
            parts.push({
              functionCall: {
                name: block.name,
                args: block.input as Record<string, any>,
                id: block.id,
              },
            });
          }
        }

        if (parts.length > 0) {
          yield {
            candidates: [
              {
                content: {
                  role: 'model',
                  parts,
                },
              },
            ],
            usageMetadata: {
              promptTokenCount,
              candidatesTokenCount,
              totalTokenCount,
            },
            text: () => currentText,
            functionCalls: () => currentToolCalls,
          } as any;
        }
      } catch (error) {
        throw new Error(
          `Anthropic generateContentStream failed: ${error instanceof Error ? error.message : String(error)}`,
        );
      }
    })();
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // rough estimate (1 token per 4 chars)
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
    throw new Error(
      'Embeddings API is not supported via Anthropic module yet.',
    );
  }
}
