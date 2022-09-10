import React, { Component } from "react";
import axios from "axios";
import { Route, Routes } from "react-router-dom";
import Feed from "./components/feed/Feed";
import About from "./components/routes/About";
import LoginForm from "./components/routes/LoginForm";
import Message from "./components/routes/Message";
import NavBar from "./components/nav/NavBar";
import RegisterForm from "./components/routes/RegisterForm";
import {getRefreshTokenIfExists, setRefreshToken, removeRefreshToken} from './utils/tokenHandler'

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
      seed: Math.random(),
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
          callback();
        })
        .catch((err) => {
          console.log(err);
        });
    }
    return false;
  };

  render() {
    return (
      <div>
        <NavBar
          title={this.state.title}
          logoutUser={this.logoutUser}
          isAuthenticated={this.isAuthenticated}
        />
        <section className="section">
          <div className="container">
            {this.state.messageType && this.state.messageText && (
              <Message
                messageType={this.state.messageType}
                messageText={this.state.messageText}
                removeMessage={this.removeMessage}
              />
            )}
            <div className="columns">
              <div className="column is-half">
                <br />
                <Routes>
                  <Route
                    exact
                    path="/feed"
                    element={
                      <Feed seed={this.state.seed}/>
                    }
                  />
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
            </div>
          </div>
        </section>
      </div>
    );
  }
}

export default App;