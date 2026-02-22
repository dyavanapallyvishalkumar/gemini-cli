/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type React from 'react';
import { Box, Text } from 'ink';
import { theme } from '../semantic-colors.js';
import { TextInput } from '../components/shared/TextInput.js';
import { useTextBuffer } from '../components/shared/text-buffer.js';
import { useUIState } from '../contexts/UIStateContext.js';

interface OllamaModelDialogProps {
  onSubmit: (model: string) => void;
  onCancel: () => void;
  error?: string | null;
  defaultValue?: string;
}

export function OllamaModelDialog({
  onSubmit,
  onCancel,
  error,
  defaultValue = 'llama3',
}: OllamaModelDialogProps): React.JSX.Element {
  const { terminalWidth } = useUIState();
  const viewportWidth = terminalWidth - 8;

  const initialModel = defaultValue;

  const buffer = useTextBuffer({
    initialText: initialModel || '',
    initialCursorOffset: initialModel?.length || 0,
    viewport: {
      width: viewportWidth,
      height: 4,
    },
    inputFilter: (text) => text.replace(/[\r\n]/g, ''),
    singleLine: true,
  });

  const handleSubmit = (value: string) => {
    onSubmit(value);
  };

  return (
    <Box
      borderStyle="round"
      borderColor={theme.border.focused}
      flexDirection="column"
      padding={1}
      width="100%"
    >
      <Text bold color={theme.text.primary}>
        Enter Ollama Model Name
      </Text>
      <Box marginTop={1} flexDirection="column">
        <Text color={theme.text.primary}>
          Please enter the name of the local Ollama model you want to use.
        </Text>
        <Text color={theme.text.secondary}>
          Make sure your Ollama daemon is running. E.g. {"'llama3'"} or{' '}
          {"'phi3'"}.
        </Text>
      </Box>
      <Box marginTop={1} flexDirection="row">
        <Box
          borderStyle="round"
          borderColor={theme.border.default}
          paddingX={1}
          flexGrow={1}
        >
          <TextInput
            buffer={buffer}
            onSubmit={handleSubmit}
            onCancel={onCancel}
            placeholder="E.g., llama3"
          />
        </Box>
      </Box>
      {error && (
        <Box marginTop={1}>
          <Text color={theme.status.error}>{error}</Text>
        </Box>
      )}
      <Box marginTop={1}>
        <Text color={theme.text.secondary}>
          (Press Enter to submit, Esc to cancel)
        </Text>
      </Box>
    </Box>
  );
}
