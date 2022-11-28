import React from "react";
import { cleanup } from "@testing-library/react";

import LoginForm from "../Routes/LoginForm";

afterEach(cleanup);

const props = {
  handleLoginFormSubmit: () => {
    return true;
  },
  isAuthenticated: () => {
    return false;
  },
};

it("renders properly", () => {
  const { getByText } = renderWithRouter(<LoginForm {...props} />);
  expect(getByText("Log In")).toHaveClass("title");
});

it("renders with default props", () => {
  const { getByLabelText, getByText } = renderWithRouter(
    <LoginForm {...props} />,
  );

  const usernameInput = getByLabelText("username");
  expect(usernameInput).not.toHaveValue();

  const passwordInput = getByLabelText("Password");
  expect(passwordInput).toHaveAttribute("type", "password");
  expect(passwordInput).not.toHaveValue();

  const buttonInput = getByText("Submit");
  expect(buttonInput).toHaveValue("Submit");
});

it("renders", () => {
  const { asFragment } = renderWithRouter(<LoginForm {...props} />);
  expect(asFragment()).toMatchSnapshot();
});
