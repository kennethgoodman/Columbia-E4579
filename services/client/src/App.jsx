import React, { useState, useContext } from "react";
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
import { DarkModeContext } from './components/darkmode/DarkModeContext';

import "./App.css";

const App = () => {
  const { darkMode } = useContext(DarkModeContext);
  const [state, setState] = useState({
    users: [],
    title: "Columbia-E4579",
    accessToken: null,
    messageType: null,
    messageText: null,
    showModal: false,
    seed: Math.random() * 180000,
  });

  const createMessage = (type, text) => {
    setState(prevState => ({
      ...prevState,
      messageType: type,
      messageText: text,
    }));
    setTimeout(() => {
      removeMessage();
    }, 3000);
  };

  const handleLoginFormSubmit = (data) => {
    const url = `${process.env.REACT_APP_API_SERVICE_URL}/auth/login`;
    axios
      .post(url, data)
      .then((res) => {
        setState(prevState => ({
          ...prevState,
          accessToken: res.data.access_token
        }));
        setRefreshToken(res.data.refresh_token);
        createMessage("success", "You have logged in successfully.");
      })
      .catch((err) => {
        console.log(err);
        createMessage("danger", "Incorrect username and/or password.");
      });
  };

  const handleRegisterFormSubmit = (data) => {
    const url = `${process.env.REACT_APP_API_SERVICE_URL}/auth/register`;
    axios
      .post(url, data)
      .then((res) => {
        setState(prevState => ({
          ...prevState,
          accessToken: res.data.access_token
        }));
        setRefreshToken(res.data.refresh_token);
        createMessage("success", "You have registered successfully.");
      })
      .catch((err) => {
        console.log(err);
        createMessage("danger", "That user already exists.");
      });
  };

  const isAuthenticated = (callback) => {
    return state.accessToken || validRefresh(callback);
  };

  const logoutUser = () => {
    removeRefreshToken();
    setState(prevState => ({
      ...prevState,
      accessToken: null
    }));
    createMessage("success", "You have logged out.");
  };

  const removeMessage = () => {
    setState(prevState => ({
      ...prevState,
      messageType: null,
      messageText: null,
    }));
  };

  const validRefresh = (callback) => {
    const token = getRefreshTokenIfExists();
    if (token) {
      axios
        .post(`${process.env.REACT_APP_API_SERVICE_URL}/auth/refresh`, {
          refresh_token: token,
        })
        .then((res) => {
          setState(prevState => ({
            ...prevState,
            accessToken: res.data.access_token
          }));
          setRefreshToken(res.data.refresh_token);
          if (callback) callback();
        })
        .catch((err) => {
          console.log(err);
        });
    }
    return false;
  };

  return (
    <div className={`App ${darkMode ? 'dark' : ''}`}>
      <NavBar
        title={state.title}
        logoutUser={logoutUser}
        isAuthenticated={isAuthenticated}
      />
      {state.messageType && state.messageText && (
        <Message
          messageType={state.messageType}
          messageText={state.messageText}
          removeMessage={removeMessage}
        />
      )}
      <Routes>
        <Route exact path="/feed" element={<Feed seed={state.seed} />} />
        <Route />
        <Route exact path="/about" element={<About />} />
        <Route
          exact
          path="/register"
          element={
            <RegisterForm
              // eslint-disable-next-line react/jsx-handler-names
              handleRegisterFormSubmit={handleRegisterFormSubmit}
              isAuthenticated={isAuthenticated}
            />
          }
        />
        <Route
          exact
          path="/login"
          element={
            <LoginForm
              // eslint-disable-next-line react/jsx-handler-names
              handleLoginFormSubmit={handleLoginFormSubmit}
              isAuthenticated={isAuthenticated}
            />
          }
        />
      </Routes>
    </div>
  );
}

export default App;
