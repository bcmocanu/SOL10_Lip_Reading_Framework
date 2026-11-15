import { Navigate, Route, Routes } from "react-router-dom";
import AppLayout from "./layout/AppLayout";
import LipReadingPage from "./pages/LipReadingPage";

export default function App() {
  return (
    <Routes>
      <Route element={<AppLayout />}>
        <Route path="/" element={<Navigate to="/processing" replace />} />
        <Route path="/processing" element={<LipReadingPage />} />
        <Route path="*" element={<Navigate to="/processing" replace />} />
      </Route>
    </Routes>
  );
}
