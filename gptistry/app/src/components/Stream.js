import { useMemo } from "react";

import Thought from "./Thought";

export default function Stream({ dispatch, root_id, pyState, ...rest }) {
  const { thoughts } = pyState;
  const stream = useMemo(() => {
    const stream = [];
    let thought = thoughts[root_id];
    while (thought) {
      thought && stream.unshift(thought);
      thought = thoughts[thought.parent_id];
    }
    return stream;
  }, [root_id, thoughts]);

  return (
    <div>
      {stream.map((thought) => (
        <Thought
          key={thought.id}
          dispatch={dispatch}
          thought_id={thought.id}
          pyState={pyState}
          {...rest}
        />
      ))}
    </div>
  );
}
