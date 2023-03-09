import {
  Streamlit,
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib";

import "./Gptistry.css";

/**
 * This is a React class component,
 * elsewhere we use the "modern" functional component.
 */
class Gptistry extends StreamlitComponentBase {
  state = { numClicks: 0, isFocused: false };

  render() {
    // Arguments that are passed to the plugin in Python are accessible
    // via `this.props.args`. Here, we access the "name" arg.
    const name = this.props.args["name"];

    // Streamlit sends us a theme object via props that we can use to ensure
    // that our component has visuals that match the active theme in a
    // streamlit app.
    const { theme } = this.props;
    const style = {};

    // Maintain compatibility with older versions of Streamlit that don't send
    // a theme object.
    if (theme) {
      // Use the theme object to style our button border. Alternatively, the
      // theme style is defined in CSS vars.
      const borderStyling = `1px solid ${
        this.state.isFocused ? theme.primaryColor : "gray"
      }`;
      style.border = borderStyling;
      style.outline = borderStyling;
    }

    // Show a button and some text.
    // When the button is clicked, we'll increment our "numClicks" state
    // variable, and send its new value back to Streamlit, where it'll
    // be available to the Python program.
    return (
      <section>
        <header className="space-y-4 rounded-md bg-slate-800 p-4 shadow-sm ring-1 ring-slate-200">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold text-slate-900 underline">
              Hello, {name}!
            </h2>
            <a
              onClick={this.onClicked}
              disabled={this.props.disabled}
              className="rounded-md bg-blue-500 py-2 pl-2 pr-3 text-sm font-medium text-white shadow-sm hover:bg-blue-400"
            >
              Click
            </a>
          </div>
        </header>
      </section>
    );
  }

  onClicked() {
    // Increment state.numClicks, and pass the new value back to
    // Streamlit via `Streamlit.setComponentValue`.
    this.setState(
      (prevState) => ({ numClicks: prevState.numClicks + 1 }),
      () => Streamlit.setComponentValue(this.state.numClicks)
    );
  }

  _onFocus() {
    this.setState({ isFocused: true });
  }

  _onBlur() {
    this.setState({ isFocused: false });
  }
}

// "withStreamlitConnection" is a wrapper function. It bootstraps the
// connection between your component and the Streamlit app, and handles
// passing arguments from Python -> Component.
//
// You don't need to edit withStreamlitConnection (but you're welcome to!).
export default withStreamlitConnection(Gptistry);
