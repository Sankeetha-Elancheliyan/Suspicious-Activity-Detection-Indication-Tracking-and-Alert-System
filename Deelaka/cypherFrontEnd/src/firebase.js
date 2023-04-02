// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyBlSfWClZlKIwztZbNWky786Hmqbvig6Kc",
  authDomain: "react-auth-1ef57.firebaseapp.com",
  projectId: "react-auth-1ef57",
  storageBucket: "react-auth-1ef57.appspot.com",
  messagingSenderId: "748563488004",
  appId: "1:748563488004:web:5312f43dd95f30407ec451"
};

// Initialize Firebase
const fb = initializeApp(firebaseConfig);

export default fb