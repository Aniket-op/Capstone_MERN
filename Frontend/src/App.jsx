import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import LoginSignupPage from "./pages/LoginSignupPage";
import Dashboard from "./pages/Dashboard";

const App = () => {
  const isAuthenticated = !!localStorage.getItem("token");
  return (
    <Router>
      <Routes>
        <Route path="/" element={isAuthenticated ? <Navigate to="/dashboard" /> : <LoginSignupPage />} />
        <Route path="/dashboard" element={isAuthenticated ? <Dashboard /> : <Navigate to="/" />} />
      </Routes>
    </Router>
  );
};

export default App;
