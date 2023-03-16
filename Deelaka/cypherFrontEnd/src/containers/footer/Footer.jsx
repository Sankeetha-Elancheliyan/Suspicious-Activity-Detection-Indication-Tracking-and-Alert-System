import React from 'react';
import './footer.css';
import storeicon from '../../assets/storeicon.png';
import followusicons from '../../assets/followusicons.png';
import logo from '../../assets/logo.png';

const Footer = () => {
  return (
    <div className='cypher__footer'>
      <div className='cypher__footer_GOP'>Get our app</div>
      <div className='cypher__footer_GOPlogo'><img src= {storeicon} alt='storeicon' /></div>
      <div className='cypher__footer_followusicons'><img src= {followusicons} alt='followusicons' /></div>
      <div className='cypher__footer-logo'>
          <img src= {logo} alt='logo' />
        </div>
        <div className='cypher__footer-links'>
        <div className='cypher__footer-menu'>
          <h4 className='menu'>Menu</h4>
          <p>home</p>
          <p>about</p>
          <p>settings</p>
          <p>profile</p>
        </div>
        <div className='cypher__footer-market'>
          <h4 className='market'>Market</h4>
          <p>upgrade</p>
          <h4>cypher+</h4>
        </div>
        <div className='cypher__footer-legal'>
          <h4 className='legal'>Legal</h4>
          <p>Genaral info</p>
          <p>Privacy and Policy</p>
          <p>Terms of Service</p>
        </div>
        <div className='followUs'><h4>Follow us</h4></div>
        </div>
        
        
      
      </div>
     
    
  )
}

export default Footer