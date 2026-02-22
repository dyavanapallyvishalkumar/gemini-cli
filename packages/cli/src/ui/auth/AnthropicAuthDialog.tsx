/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import type React from 'react';
import { useRef, useEffect } from 'react';
import { Box, Text } from 'ink';
import { theme } from '../semantic-colors.js';
import { TextInput } from '../components/shared/TextInput.js';
import { useTextBuffer } from '../components/shared/text-buffer.js';
import { useUIState } from '../contexts/UIStateContext.js';
import { clearAnthropicApiKey, debugLogger } from '@google/gemini-cli-core';
import { useKeypress } from '../hooks/useKeypress.js';
import { keyMatchers, Command } from '../keyMatchers.js';

interface AnthropicAuthDialogProps {
  onSubmit: (apiKey: string) => void;
  onCancel: () => void;
  error?: string | null;
  defaultValue?: string;
}

export function AnthropicAuthDialog({
  onSubmit,
  onCancel,
  error,
  defaultValue = '',
}: AnthropicAuthDialogProps): React.JSX.Element {
  const { terminalWidth } = useUIState();
  const viewportWidth = terminalWidth - 8;

  const pendingPromise = useRef<{ cancel: () => void } | null>(null);

  useEffect(
    () => () => {
      pendingPromise.current?.cancel();
    },
    [],
  );

  const initialApiKey = defaultValue;

  const buffer = useTextBuffer({
    initialText: initialApiKey || '',
    initialCursorOffset: initialApiKey?.length || 0,
    viewport: {
      width: viewportWidth,
      height: 4,
    },
    inputFilter: (text) =>
      text.replace(/[^a-zA-Z0-9_-]/g, '').replace(/[\r\n]/g, ''),
    singleLine: true,
  });

  const handleSubmit = (value: string) => {
    onSubmit(value);
  };

  const handleClear = () => {
    pendingPromise.current?.cancel();

    let isCancelled = false;
    const wrappedPromise = new Promise<void>((resolve, reject) => {
      clearAnthropicApiKey().then(
        () => !isCancelled && resolve(),
        () => !isCancelled && reject(new Error('Failed to clear key')),
      );
    });

    pendingPromise.current = {
      cancel: () => {
        isCancelled = true;
      },
    };

    return wrappedPromise
      .then(() => {
        buffer.setText('');
      })
      .catch((err: unknown) => {
        debugLogger.debug('Failed to clear Anthropic API key:', err);
      });
  };

  useKeypress(
    (key) => {
      if (keyMatchers[Command.CLEAR_INPUT](key)) {
        void handleClear();
        return true;
      }
      return false;
    },
    { isActive: true },
  );

  return (
    <Box
      borderStyle="round"
      borderColor={theme.border.focused}
      flexDirection="column"
      padding={1}
      width="100%"
    >
      <Text bold color={theme.text.primary}>
        Enter Anthropic API Key
      </Text>
      <Box marginTop={1} flexDirection="column">
        <Text color={theme.text.primary}>
          Please enter your Anthropic API key. It will be securely stored in
          your system keychain.
        </Text>
        <Text color={theme.text.secondary}>
          You can get an API key from{' '}
          <Text color={theme.text.link}>
            https://console.anthropic.com/settings/keys
          </Text>
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
            placeholder="Paste your Anthropic API key here"
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
          (Press Enter to submit, Esc to cancel, Ctrl+C to clear stored key)
        </Text>
      </Box>
    </Box>
  );
}
