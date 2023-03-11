import { useState, useEffect } from "react";
import CodeEditor from "./CodeEditor";
import { Button, Box } from "@chakra-ui/react";
import { BeatLoader } from "react-spinners";

const c =
  "search index=redteam_50k RED=1 | Table src_computer, dst_computer, time, eventtype, success_or_failure";

const ActionButton = ({ children, spinning, ...props }) => (
  <Button
    spinner={spinning ? <BeatLoader size={8} color="white" /> : undefined}
    {...props}
  >
    {children}
  </Button>
);

export default function Thought({
  dispatcher,
  pyState: { thoughts, busy },
  thought_id,
  updateTime,
}) {
  const thought = thoughts[thought_id];
  const [code, setCode] = useState(thought.code);

  useEffect(() => {
    // hm, what if another component updates the code?
    // and this one has unsaved changes :(
    // TODO: save on blur.

    if (thought.updated_at) {
      setCode(thought.code);
    }
  }, [thought.updated_at]);

  return (
    <Box>
      <CodeEditor code={code} onChange={(v) => setCode(v)} />

      <ActionButton
        onClick={() => dispatcher.fix({ ...thought, code })}
        onBlur={() => dispatcher.update({ ...thought, code })}
        spinning={busy}
      >
        Fix
      </ActionButton>
    </Box>
  );
}
