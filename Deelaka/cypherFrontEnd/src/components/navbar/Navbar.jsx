import React from 'react';
import {RiMenu3Line, RiCloseLin} from 'react-icons/ri';
import { useNavigate } from 'react-router-dom';
import logo from '../../assets/logo.png';
import './navbar.css';


const Navbar = () => {
  const navigate = useNavigate();
  function handleClick (){
    navigate("/profile")
  }

  function clickHome (){
    navigate("/home")
  }
  
  return (
    <div className='cypher__navbar'>
      <div className='cypher__navbar-links'>
        <div className='cypher__navbar-links_logo'>
          <img src= {logo} alt='logo' />
        </div>
        <div className='cypher__navbar-links_container'>
          <p onClick={(e)=>clickHome()}>home</p>
          <p><a href='#notifications'>notifications</a></p>
          <p><a href='#about'>about</a></p>
          <p><a href='#settings'>settings</a></p>
          <p onClick={(e)=>handleClick()}>profile</p>
        </div>
      </div>
    </div>
  )
}

export default Navbar