import { useMemo } from "react";

import { IconButton, Box } from "@chakra-ui/react";
import { ArrowForwardIcon } from "@chakra-ui/icons";

export default function ThoughtHistory({
  dispatcher,
  root_id,
  pyState,
  ...rest
}) {
  const { thoughts } = pyState;
  const history = useMemo(() => {
    const history = Object.values(thoughts);
    history.sort((a, b) => a.updated_at - b.updated_at);
    return history;
  }, [root_id, thoughts]);

  return (
    <Box>
      {history.map((thought) => (
        <Box
          key={thought.id}
          display="flex"
          justifyContent="space-between"
          alignItems="center"
        >
          {thought.id} {thought.intent}
          <IconButton
            isDisabled={thought.id === root_id}
            size="sm"
            onClick={() => dispatcher.focus(thought.id)}
            aria-label="Search database"
            icon={<ArrowForwardIcon />}
          />
        </Box>
      ))}
    </Box>
  );
}
