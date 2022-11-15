import React, { Component } from "react";
import axios from "axios";
import { Route, Routes } from "react-router-dom";
import Feed from "./components/Feed";
import About from "./components/Routes/About";
import LoginForm from "./components/Routes/LoginForm";
import Message from "./components/Routes/Message";
import NavBar from "./components/Navbar";
import RegisterForm from "./components/Routes/RegisterForm";
import {
  getRefreshTokenIfExists,
  setRefreshToken,
  removeRefreshToken,
} from "./utils/tokenHandler";

import "./App.css";

class App extends Component {
  constructor() {
    super();

    this.state = {
      users: [],
      title: "Columbia-E4579",
      accessToken: null,
      messageType: null,
      messageText: null,
      showModal: false,
      seed: Math.random() * 1000000,
    };
  }

  createMessage = (type, text) => {
    this.setState({
      messageType: type,
      messageText: text,
    });
    setTimeout(() => {
      this.removeMessage();
    }, 3000);
  };

  handleLoginFormSubmit = (data) => {
    const url = `${process.env.REACT_APP_API_SERVICE_URL}/auth/login`;
    axios
      .post(url, data)
      .then((res) => {
        this.setState({ accessToken: res.data.access_token });
        setRefreshToken(res.data.refresh_token);
        this.createMessage("success", "You have logged in successfully.");
      })
      .catch((err) => {
        console.log(err);
        this.createMessage("danger", "Incorrect username and/or password.");
      });
  };

  handleRegisterFormSubmit = (data) => {
    const url = `${process.env.REACT_APP_API_SERVICE_URL}/auth/register`;
    axios
      .post(url, data)
      .then((res) => {
        this.setState({ accessToken: res.data.access_token });
        setRefreshToken(res.data.refresh_token);
        this.createMessage("success", "You have logged in successfully.");
        this.createMessage("success", "You have registered successfully.");
      })
      .catch((err) => {
        console.log(err);
        this.createMessage("danger", "That user already exists.");
      });
  };

  isAuthenticated = (callback) => {
    return this.state.accessToken || this.validRefresh(callback);
  };

  logoutUser = () => {
    removeRefreshToken();
    this.setState({ accessToken: null });
    this.createMessage("success", "You have logged out.");
  };

  removeMessage = () => {
    this.setState({
      messageType: null,
      messageText: null,
    });
  };

  validRefresh = (callback) => {
    const token = getRefreshTokenIfExists();
    if (token) {
      axios
        .post(`${process.env.REACT_APP_API_SERVICE_URL}/auth/refresh`, {
          refresh_token: token,
        })
        .then((res) => {
          this.setState({ accessToken: res.data.access_token });
          setRefreshToken(res.data.refresh_token);
          if (callback) callback();
        })
        .catch((err) => {
          console.log(err);
        });
    }
    return false;
  };

  render() {
    return (
      <div className="App">
        <NavBar
          title={this.state.title}
          logoutUser={this.logoutUser}
          isAuthenticated={this.isAuthenticated}
        />
        {this.state.messageType && this.state.messageText && (
          <Message
            messageType={this.state.messageType}
            messageText={this.state.messageText}
            removeMessage={this.removeMessage}
          />
        )}
        <Routes>
          <Route exact path="/feed" element={<Feed seed={this.state.seed} />} />
          <Route />
          <Route exact path="/about" element={<About />} />
          <Route
            exact
            path="/register"
            element={
              <RegisterForm
                // eslint-disable-next-line react/jsx-handler-names
                handleRegisterFormSubmit={this.handleRegisterFormSubmit}
                isAuthenticated={this.isAuthenticated}
              />
            }
          />
          <Route
            exact
            path="/login"
            element={
              <LoginForm
                // eslint-disable-next-line react/jsx-handler-names
                handleLoginFormSubmit={this.handleLoginFormSubmit}
                isAuthenticated={this.isAuthenticated}
              />
            }
          />
        </Routes>
      </div>
    );
  }
}

export default App;
