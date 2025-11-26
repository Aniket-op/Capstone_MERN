const ChartCard = ({ title, subtitle, children }) => {
  return (
    <div className="bg-white shadow-md rounded-xl p-6">
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-gray-900 mb-1">{title}</h2>
        {subtitle && <p className="text-sm text-gray-500">{subtitle}</p>}
      </div>
      {children}
    </div>
  );
};

export default ChartCard;
