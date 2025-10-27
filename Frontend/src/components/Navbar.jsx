const Navbar = ({ onNotificationClick }) => {
  const handleLogout = () => {
    localStorage.removeItem("token");
    window.location.href = "/";
  };

  return (
    <nav className="bg-gray-800 text-white p-4 flex justify-between items-center">
      <h1 className="text-lg font-bold">Smart Solar Panel Dashboard</h1>

      <div className="flex items-center gap-4">
        <button
          onClick={onNotificationClick}
          className="bg-blue-500 hover:bg-blue-600 px-3 py-1 rounded"
        >
          Notifications
        </button>
        <button
          onClick={handleLogout}
          className="bg-red-500 hover:bg-red-600 px-3 py-1 rounded"
        >
          Logout
        </button>
      </div>
    </nav>
  );
};

export default Navbar;
