const ChartCard = ({ title, children }) => {
  return (
    <div className="bg-white shadow-md rounded p-4 mb-4">
      <h2 className="font-semibold text-gray-500 mb-2">{title}</h2>
      {children}
    </div>
  );
};

export default ChartCard;
