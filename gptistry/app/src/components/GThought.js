// import { useEffect } from "react";
import CodeEditor from "./CodeEditor";

const c =
  "****splunk: | search index=redteam_50k RED=1 | Table src_computer, dst_computer, time, eventtype, success_or_failure";

export default function GThought({ aiState, dispatch, actions }) {
  const actionSet = new Set(actions);

  if (!aiState) return <div>Unknown</div>;
  return (
    <div>
      <CodeEditor code={c} />
      {actionSet.has("fix") && (
        <button onClick={() => dispatch("fix", [])}>FIX</button>
      )}
    </div>
  );
}
