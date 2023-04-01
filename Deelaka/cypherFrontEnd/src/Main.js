import React from 'react'
import App from './App';
import Login from "./Login";
import Profile from "./Profile";
import { BrowserRouter, Route, Router, Routes } from 'react-router-dom';

///db connect
function Main() {
    return (
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Login />} />
          <Route path="home/*" element={<App />} />
          <Route path="profile/*" element={<Profile />} />
        </Routes>
      </BrowserRouter>
    );
  }

export default Main