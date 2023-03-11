import { isEqual } from "lodash";
import { useState } from "react";

function useMemoizedState(initialValue) {
  const [state, _setState] = useState(initialValue);

  const setState = (newState) => {
    _setState((prev) => {
      if (!isEqual(newState, prev)) {
        return newState;
      } else {
        return prev;
      }
    });
  };

  return [state, setState];
}

export default useMemoizedState;
