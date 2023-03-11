export default class Dispatcher {
  constructor(dispatch) {
    this.dispatch = dispatch;
  }

  fix(thought) {
    this.dispatch({ type: "fix", thought });
  }
  update(thought) {
    this.dispatch({ type: "update", thought });
  }
  focus(thought_id) {
    this.dispatch({ type: "focus", thought_id });
  }
}
