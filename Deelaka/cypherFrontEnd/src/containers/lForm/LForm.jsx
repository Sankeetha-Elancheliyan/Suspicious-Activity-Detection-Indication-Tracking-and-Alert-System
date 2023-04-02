import React, { useState } from 'react'
import './lForm.css';
import logo from '../../assets/logo.png';
import storeicon from '../../assets/storeicon.png';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import fb from '../../firebase';
import { getAuth, signInWithEmailAndPassword } from "firebase/auth";

const LForm = () => {


  
const navigate = useNavigate();
const auth = getAuth(fb);

const[email, setEmail] = useState("")
const[password, setPassword] = useState("")

const signIn = () => {
signInWithEmailAndPassword(auth, email, password)
  .then((userCredential) => {
    // Signed in 
    const user = userCredential.user;
    console.log(user);
    navigate("/home")
    // ...
  })
  .catch((error) => {
    const errorCode = error.code;
    const errorMessage = error.message;
    alert(errorCode)
  });
}



  
  return (
    <div className='l'>
        <div className='form_logo'><img src= {logo} alt='logo' /></div>
         <div className='cover'>
            <h1 className='signin'>Sign in</h1>
            <input className='box' type= "email" placeholder='email' onChange={(e) => setEmail(e.target.value)}></input>
            <input className='box' type="password" placeholder='Password' onChange={(e) => setPassword(e.target.value)}></input>
             <div className='login-btn' onClick={signIn}>Login</div>
             <h1 className='form-GOA'>Get our app</h1>
             <div className='form_store'><img src= {storeicon} alt='' /></div>


         </div>
        
    
    
    </div>
  )
}

export default LForm