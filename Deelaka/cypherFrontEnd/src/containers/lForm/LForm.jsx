import React from 'react'
import './lForm.css';
import logo from '../../assets/logo.png';
import storeicon from '../../assets/storeicon.png';

const LForm = () => {
  return (
    <div className='l'>
        <div className='form_logo'><img src= {logo} alt='logo' /></div>
         <div className='cover'>
            <h1 className='signin'>Sign in</h1>
            <input className='box' type= "text" placeholder='user ID'></input>
            <input className='box' type= "text" placeholder='Password'></input>
             <div className='login-btn'>Login</div>
             <h1 className='form-GOA'>Get our app</h1>
             <div className='form_store'><img src= {storeicon} alt='' /></div>


         </div>
        
    
    
    </div>
  )
}

export default LForm