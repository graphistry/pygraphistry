import React, { useEffect } from "react";
import Prism from "prismjs";

// use prism-react-renderer?

export default function SplunkEditor({ code, onChange }) {
  const handleKeyDown = (e) => {
    let value = code;
    let selStartPos = e.currentTarget.selectionStart;

    // handle 4-space indent on
    if (e.key === "Tab") {
      value =
        value.substring(0, selStartPos) +
        "    " +
        value.substring(selStartPos, value.length);
      e.currentTarget.selectionStart = selStartPos + 3;
      e.currentTarget.selectionEnd = selStartPos + 4;
      e.preventDefault();

      onChange(value);
    }
  };

  useEffect(() => {
    Prism.highlightAll();
  }, [code]);

  return (
    <div className="code-edit-container">
      <textarea
        className="code-input"
        value={code}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
      ></textarea>
      <pre className="code-output">
        <code className={`language-splunk-spl`}>{code}</code>
      </pre>
    </div>
  );
}
