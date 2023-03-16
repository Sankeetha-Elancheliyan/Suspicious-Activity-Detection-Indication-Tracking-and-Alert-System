import React from 'react';
import './Profile.css';
import logo from './assets/logo.png';
import profileLogo from './assets/profileLogo.png';
import storeicon from './assets/storeicon.png';
import { Navbar } from './components';


const Profile = () => {
  return (
    
    <div className='profile'>
      <Navbar/>
            <div className='center'>
          <div className='profile_logo'><img src= {profileLogo} alt='profileLogo' /></div>
          <h1 className='name'>Profile</h1>
          <div className='settings_btn'>SETTINGS</div> 
          <div className='logout_btn'>LOGOUT</div> 
          <div className='lockdown_btn'>LOCKDOWN</div> 
          <h1 className='profile_GOA'>Get our app</h1>
          <div className='profile_store'><img src= {storeicon} alt='' /></div>         
        </div>
        
    </div>
  )
}

export default Profile