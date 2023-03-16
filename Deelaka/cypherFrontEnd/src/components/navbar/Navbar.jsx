import React from 'react';
import {RiMenu3Line, RiCloseLin} from 'react-icons/ri';
import logo from '../../assets/logo.png';
import './navbar.css';

const Navbar = () => {
  return (
    <div className='cypher__navbar'>
      <div className='cypher__navbar-links'>
        <div className='cypher__navbar-links_logo'>
          <img src= {logo} alt='logo' />
        </div>
        <div className='cypher__navbar-links_container'>
          <p><a href='#home'>home</a></p>
          <p><a href='#notifications'>notifications</a></p>
          <p><a href='#about'>about</a></p>
          <p><a href='#settings'>settings</a></p>
          <p><a href='#profile'>profile</a></p>
        </div>
      </div>
    </div>
  )
}

export default Navbar