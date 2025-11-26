import { useState } from "react";
import axios from "axios";
import { Zap } from "lucide-react";

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
      if (data.username) {
        localStorage.setItem("username", data.username);
      }
      window.location.href = "/dashboard";
    } catch (err) {
      alert(err.response.data.message);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-slate-100 w-full h-screen">
      <div className="bg-white rounded-2xl shadow-xl w-full max-w-md p-10">
        {/* Logo */}
        <div className="flex justify-center mb-6">
          <div className="w-16 h-16 bg-blue-600 rounded-full flex items-center justify-center">
            <Zap className="w-8 h-8 text-white" />
          </div>
        </div>

        {/* Title */}
        <h1 className="text-3xl font-bold text-gray-900 text-center mb-2">
          Smart Solar Panel Cleaner
        </h1>

        {/* Subtitle */}
        <p className="text-gray-600 text-center mb-8 text-base">
          Monitor and optimize your solar panel performance
        </p>

        {/* Tabs */}
        <div className="flex mb-8 border-b border-gray-200">
          <button
            type="button"
            onClick={() => setIsLogin(true)}
            className={`flex-1 py-3 text-center font-medium ${
              isLogin
                ? "bg-white border border-gray-200 border-b-white -mb-px text-blue-600 rounded-t-lg"
                : "text-gray-700 hover:text-gray-900"
            }`}
          >
            Sign In
          </button>
          <button
            type="button"
            onClick={() => setIsLogin(false)}
            className={`flex-1 py-3 text-center font-medium ${
              !isLogin
                ? "bg-white border border-gray-200 border-b-white -mb-px text-blue-600 rounded-t-lg"
                : "text-gray-700 hover:text-gray-900"
            }`}
          >
            Sign Up
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-5">
          {!isLogin && (
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-gray-700 mb-2">
                Username
              </label>
              <input
                type="text"
                id="username"
                name="username"
                placeholder="Choose a username"
                value={formData.username}
                onChange={handleChange}
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                required={!isLogin}
              />
            </div>
          )}

          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-2">
              Email address
            </label>
            <input
              type="email"
              id="email"
              name="email"
              placeholder={isLogin ? "Enter your email" : "Enter your email"}
              value={formData.email}
              onChange={handleChange}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              required
            />
          </div>

          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-2">
              Password
            </label>
            <input
              type="password"
              id="password"
              name="password"
              placeholder={isLogin ? "Enter your password" : "Create a password"}
              value={formData.password}
              onChange={handleChange}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              required
            />
          </div>

          <button
            type="submit"
            className={`w-full py-3 rounded-lg text-white font-medium ${
              isLogin
                ? "bg-blue-600 hover:bg-blue-700"
                : "bg-green-600 hover:bg-green-700"
            }`}
          >
            {isLogin ? "Sign In" : "Create Account"}
          </button>
        </form>
      </div>
    </div>
  );
};

export default LoginSignupPage;
