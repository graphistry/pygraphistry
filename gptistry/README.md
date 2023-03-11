# GPT prototyping for Graphistry

### Getting started

- `npm i`
- `npm start`

### How to build for production

- WIP
- `npm run build`

### How this was created

Based off of the streamlit [component-template](https://github.com/streamlit/component-template). But then starting fresh with [Create React App](https://github.com/facebook/create-react-app) and using js instead of ts.

[Components from ChakraUI](https://chakra-ui.com)

Also using eslint and prettier as [described here](https://medium.com/how-to-react/config-eslint-and-prettier-in-visual-studio-code-for-react-js-development-97bb2236b31a).

Streamlit's template is using the Apache License 2.0

### eslint and prettier

For vscode, using:

```
    "editor.codeActionsOnSave": {
        "source.fixAll.eslint": true
    },
    "editor.formatOnSave": true,
    "eslint.alwaysShowStatus": true
```

## Things to improve or fix

#### Compatibility react-scripts 5

https://discuss.streamlit.io/t/apache-arrow-module-not-found-error/21952/4
Breaks tailwind easy install, so:

#### Use rewire to override webpack configuration

https://stackoverflow.com/questions/64002604/how-to-make-create-react-app-support-mjs-files-with-webpack
