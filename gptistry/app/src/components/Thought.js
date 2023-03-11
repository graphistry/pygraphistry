import { useState, useEffect } from "react";
import CodeEditor from "./CodeEditor";
import { Button, Box, Input, Textarea, IconButton } from "@chakra-ui/react";
import { BeatLoader } from "react-spinners";
import { SunIcon, ChatIcon } from "@chakra-ui/icons";

// search index=redteam_50k RED=1 | Table src_computer, dst_computer, time, eventtype, success_or_failure

const withSpinner = (spinning) =>
  spinning ? { spinner: <BeatLoader size={8} color="white" /> } : undefined;

export default function Thought({
  dispatcher,
  pyState: { thoughts, busy },
  thought_id,
  updateTime,
}) {
  const thought = thoughts[thought_id];
  const [code, setCode] = useState(thought.code);
  const [intent, setIntent] = useState(thought.intent);
  const [prompt, setPrompt] = useState(thought.prompt);

  useEffect(() => {
    if (thought.updated_at) {
      setCode(thought.code);
      setIntent(thought.intent);
      setPrompt(thought.prompt);
    }
  }, [thought.updated_at]);

  const currentThought = () => ({
    ...thought,
    code,
    intent,
    prompt,
    fresh: false,
  });

  const onBlur = () => {
    if (
      thought.code !== code ||
      thought.intent !== intent ||
      thought.prompt !== prompt
    ) {
      setTimeout(() => {
        // TODO: this is a total hack to get around the siutaiton where a user
        // blurs the input and then clicks a button, we want the button action
        // to get precendence and the event loop might swallow events :/
        dispatcher.update(currentThought());
      }, 50);
    }
  };

  return (
    <Box
      display="flex"
      flexDirection="column"
      gap={2}
      p={3}
      rounded="md"
      bgGradient="linear(to-l, HSLA(198, 50%, 97%, 0.05), HSLA(173, 53%, 89%, 0.05))"
    >
      <Input
        variant="flushed"
        size="sm"
        placeholder="intent"
        value={intent}
        onChange={(v) => setIntent(v.target.value)}
        onBlur={onBlur}
      />

      <Textarea
        value={prompt}
        onChange={(v) => setPrompt(v.target.value)}
        size="md"
        onBlur={onBlur}
      />

      <CodeEditor
        placeholder="code"
        code={code}
        onChange={(v) => setCode(v)}
        onBlur={onBlur}
      />

      <Box display="flex" justifyContent="right" gap="1em">
        <IconButton
          size="sm"
          disabled={!intent || !prompt}
          onClick={() => dispatcher.ask(currentThought())}
          {...withSpinner(busy)}
          aria-label="Ask"
          icon={<ChatIcon />}
        />
        {code && (
          <Button
            size="sm"
            onClick={() => dispatcher.fix(currentThought())}
            {...withSpinner(busy)}
          >
            Fix
          </Button>
        )}
        {code && (
          <IconButton
            size="sm"
            onClick={() => dispatcher.plot(currentThought())}
            {...withSpinner(busy)}
            aria-label="Plot"
            icon={<SunIcon />}
          />
        )}
      </Box>
    </Box>
  );
}
