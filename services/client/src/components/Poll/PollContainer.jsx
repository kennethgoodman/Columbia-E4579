import React, { useEffect, useState } from "react";
import axios from "axios";
import { getRefreshTokenIfExists } from "../../utils/tokenHandler";
import Poll from "./Poll";
import AdminPoll from "./AdminPoll"; 
import { useNavigate } from 'react-router-dom'; 

const PollContainer = () => {
  const [isAdmin, setIsAdmin] = useState(false);
  const [shouldRedirect, setShouldRedirect] = useState(false);
  const navigate = useNavigate();
  if (!getRefreshTokenIfExists() || shouldRedirect) {
    // If not authenticated, redirect to login
    navigate('/login');
    return
  }

  useEffect(() => {
    const checkAdmin = async () => {
      try {
        const response = await axios.get(`${process.env.REACT_APP_API_SERVICE_URL}/auth/status`, {
          headers: {
            Authorization: `Bearer ${getRefreshTokenIfExists()}`
          }
        });
        setIsAdmin(response.data.is_admin); 
      } catch (error) {
        navigate('/login');
      }
    };

    checkAdmin();
  }, [navigate]);

  return <div> {isAdmin ? <AdminPoll /> : <Poll />}</div>;
};

export default PollContainer;