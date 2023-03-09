import React from "react";
import ReactDOM from "react-dom/client";
import "./index.css";
import Gptistry from "./Gptistry";
import reportWebVitals from "./reportWebVitals";

console.log = (...args) => console.log("@Gptistry", ...args);

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <Gptistry />
  </React.StrictMode>
);

reportWebVitals();
