import { render, screen } from "@testing-library/react";
import Gptistry from "./Gptistry";

test("renders learn react link", () => {
  render(<Gptistry />);
  const linkElement = screen.getByText(/learn react/i);
  expect(linkElement).toBeInTheDocument();
});
