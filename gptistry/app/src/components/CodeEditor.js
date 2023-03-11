import React, { useEffect, useRef } from "react";
import Prism from "prismjs";
import "prismjs/components/prism-splunk-spl";

import "./CodeEditor.css";

// Reference implementations:
// https://css-tricks.com/creating-an-editable-textarea-that-supports-syntax-highlighted-code/
// https://github.com/JCGuest/codepen/blob/master/src/components/CodeEditor.js

export default function CodeEditor({
  code,
  onChange,
  language = "splunk-spl",
}) {
  const textArea = useRef(null);
  const codeBlock = useRef(null);

  const syncScroll = () => {
    codeBlock.current.scrollTop = textArea.current.scrollTop;
    codeBlock.current.scrollLeft = textArea.current.scrollLeft;
  };

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
    setTimeout(() => syncScroll(), 100);
  }, [code]);

  // The pre tag ignores a trailing newline, so add a space to compensate.
  const fixedCode = code.endsWith("\n") ? code + " " : code;

  return (
    <div className="code-edit-container">
      <textarea
        id="code-editor"
        value={code}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        ref={textArea}
        onScroll={syncScroll}
      ></textarea>
      <pre id="code-highlighter" ref={codeBlock}>
        <code className={`language-${language}`}>{fixedCode}</code>
      </pre>
    </div>
  );
}
