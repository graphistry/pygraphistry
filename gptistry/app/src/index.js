import { useState, useEffect, StrictMode, useMemo } from "react";
import { Streamlit } from "streamlit-component-lib";
import ReactDOM from "react-dom/client";
import reportWebVitals from "./reportWebVitals";
import {
  ChakraProvider,
  extendTheme,
  withDefaultColorScheme,
} from "@chakra-ui/react";
import tinycolor from "tinycolor2";
import { isEqual } from "lodash";

import Dispatcher from "./Dispatcher";
import useMemoizedState from "./hooks/useMemoizedState";
import Plot from "./components/Plot";
import Stream from "./components/Stream";
import Thought from "./components/Thought";
import ThoughtHistory from "./components/ThoughtHistory";

import "./index.css";

const DEFAULT_STREAMLIT_THEME = {
  primaryColor: "#ff4b4b",
  backgroundColor: "#262730",
  secondaryBackgroundColor: "#0e1117",
  textColor: "#fafafa",
  base: "dark",
  font: '"Source Sans Pro", sans-serif',
  linkText: "hsla(209, 100%, 59%, 1)",
  fadedText05: "rgba(250, 250, 250, 0.1)",
  fadedText10: "rgba(250, 250, 250, 0.2)",
  fadedText20: "rgba(250, 250, 250, 0.3)",
  fadedText40: "rgba(250, 250, 250, 0.4)",
  fadedText60: "rgba(250, 250, 250, 0.6)",
  bgMix: "rgba(26, 28, 36, 1)",
  darkenedBgMix100: "hsla(228, 16%, 72%, 1)",
  darkenedBgMix25: "rgba(172, 177, 195, 0.25)",
  darkenedBgMix15: "rgba(172, 177, 195, 0.15)",
  lightenedBg05: "hsla(234, 12%, 19%, 1)",
};

const EXAMPLE_PROMPT = `
write a splunk query for the index \`redteam_50k\` that uses the src and dst information
to output a table for events where red=1
`;

const LOADING = "loading";

const componentByType = {
  plot: Plot,
  stream: Stream,
  thought: Thought,
  history: ThoughtHistory,
  [LOADING]: () => <div>Loading...</div>,
  debug: (props) => <pre>Status: {JSON.stringify(props, null, 2)}</pre>,
};

// based on: ./node_modules/streamlit-component-lib/dist/StreamlitReact.js
// TODO: withStreamlitConnection creates an error boundary, do this as well.
export function App() {
  // For debugging
  const [stData, setStData] = useState();

  const [type, setType] = useMemoizedState(LOADING);
  const [pyState, setPyState] = useMemoizedState();
  const [componentProps, setComponentProps] = useMemoizedState();
  const [key, setKey] = useMemoizedState("unset");
  const [updateTime, setUpdateTime] = useMemoizedState(Date.now());

  const [theme, setTheme] = useMemoizedState();
  const [disabled, setDisabled] = useMemoizedState();

  console.log(`Render ${type}-${key}`, {
    stData,
    pyState,
    componentProps,
    theme,
    disabled,
  });

  const dispatcher = useMemo(
    () =>
      new Dispatcher((args) => {
        console.log("Dispatching action", args);
        Streamlit.setComponentValue({
          ...args,
          timestamp: Date.now(),
        });
      }),
    []
  );

  useEffect(() => {
    const cb = (event) => {
      const { args = {}, theme = {}, disabled = false } = event.detail ?? {};
      setStData(event.detail);

      const stheme = {
        ...DEFAULT_STREAMLIT_THEME,
        ...(theme ?? {}),
      };
      setTheme(
        extendTheme(
          {
            colors: {
              graphistry: {
                50: "#D6F3FF",
                100: "#A5E4FF",
                200: "#7DD3F8",
                300: "#61BEE4",
                400: "#46A6D0",
                500: "#3090B8",
                600: "#20759A",
                700: "#185874",
                800: "#103C4F",
                900: "#082029",
              },
            },
            initialColorMode: tinycolor(stheme.backgroundColor).isDark()
              ? "dark"
              : "light",
            useSystemColorMode: false,
          },
          withDefaultColorScheme({ colorScheme: "graphistry" })
        )
      );
      setDisabled(disabled);

      const {
        type = LOADING,
        state = {},
        key = "unset",
        ...componentProps
      } = args;
      setType(type);
      setPyState(state);
      setComponentProps(componentProps);
      setKey(key);
      if (!isEqual(state, pyState)) {
        setUpdateTime(Date.now());
      }
    };
    Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, cb);
    Streamlit.setComponentReady();
    return () =>
      Streamlit.events.removeEventListener(Streamlit.RENDER_EVENT, cb);
  }, []);

  useEffect(() => {
    if (stData) {
      Streamlit.setFrameHeight();
    }
  }, [stData]);

  const Component = componentByType[type];
  return (
    <ChakraProvider theme={theme}>
      <Component
        disabled={disabled}
        key={key}
        dispatcher={dispatcher}
        pyState={pyState}
        updateTime={updateTime}
        {...componentProps}
      />
    </ChakraProvider>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <StrictMode>
    <App />
  </StrictMode>
);

reportWebVitals();
