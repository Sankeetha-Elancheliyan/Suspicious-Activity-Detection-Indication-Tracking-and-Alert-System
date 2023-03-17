import React from 'react';
import './header.css';

const Header = () => {
  return (
    <div className='screen'>
      <select className='dropdown'>
        <option value="fruit">Violence Detection</option>
        <option value="vegetable">Intrusion Detection</option>
        <option value="meat">Person re-identification</option>

      </select>
    </div>
    
  )
} 

export default Header

