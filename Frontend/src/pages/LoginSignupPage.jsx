import { useState } from "react";
import axios from "axios";

const LoginSignupPage = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({ username: "", email: "", password: "" });

  const handleChange = (e) => setFormData({ ...formData, [e.target.name]: e.target.value });

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const url = isLogin ? "http://localhost:5000/api/auth/login" : "http://localhost:5000/api/auth/signup";
      const { data } = await axios.post(url, formData);
      localStorage.setItem("token", data.token);
      window.location.href = "/dashboard";
    } catch (err) {
      alert(err.response.data.message);
    }
  };

  return (
    <div className="flex items-center justify-center h-screen bg-gray-100">
      <form className="bg-white p-6 rounded shadow-md w-96" onSubmit={handleSubmit}>
        {!isLogin && (
          <input
            type="text"
            name="username"
            placeholder="Username"
            value={formData.username}
            onChange={handleChange}
            className="w-full p-2 border rounded mb-3"
          />
        )}
        <input
          type="email"
          name="email"
          placeholder="Email"
          value={formData.email}
          onChange={handleChange}
          className="w-full p-2 border rounded mb-3"
        />
        <input
          type="password"
          name="password"
          placeholder="Password"
          value={formData.password}
          onChange={handleChange}
          className="w-full p-2 border rounded mb-3"
        />
        <button type="submit" className="w-full bg-blue-500 text-white p-2 rounded">
          {isLogin ? "Login" : "Signup"}
        </button>
        <p className="mt-3 text-center text-sm">
          {isLogin ? "Don't have an account?" : "Already have an account?"}{" "}
          <span className="text-blue-500 cursor-pointer" onClick={() => setIsLogin(!isLogin)}>
            {isLogin ? "Signup" : "Login"}
          </span>
        </p>
      </form>
    </div>
  );
};

export default LoginSignupPage;
