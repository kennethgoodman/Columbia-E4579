import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import { getRefreshTokenIfExists } from '../../utils/tokenHandler';
import './Poll.css'; 
import { useContext } from 'react';
import { DarkModeContext } from '../../components/darkmode/DarkModeContext';


const Poll = () => {
  const [poll, setPoll] = useState(null);
  const [selectedOption, setSelectedOption] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [successMessage, setSuccessMessage] = useState(null);
  const navigate = useNavigate(); 
  const { darkMode } = useContext(DarkModeContext);

  useEffect(() => {
    const fetchPoll = async () => {
      try {
        const response = await axios.get(`${process.env.REACT_APP_API_SERVICE_URL}/polls`, { // Get the first available poll
          headers: {
            Authorization: `Bearer ${getRefreshTokenIfExists()}`
          }
        });

        if (response.data.length > 0) { 
          setPoll(response.data[0]); 
        } else {
          setErrorMessage('No available polls at this time.');
        }

      } catch (error) {
        if (error.response && error.response.status === 401) {
          // Handle unauthorized error (e.g., token expired)
          navigate('/login'); // Redirect to login 
        } else {
          setErrorMessage('Error loading poll. Please try again later.');
          console.error('Error fetching poll:', error);
        }
      }
    };

    fetchPoll();
  }, [navigate]); // Add navigate to dependency array

  const handleOptionChange = (event) => {
    setSelectedOption(parseInt(event.target.value, 10)); // Parse value as integer
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const response = await axios.post(
        `${process.env.REACT_APP_API_SERVICE_URL}/polls/${poll.id}/votes`,
        { choice: selectedOption },
        {
          headers: {
            Authorization: `Bearer ${getRefreshTokenIfExists()}`,
            'Content-Type': 'application/json',
          },
        }
      );

      if (response.status === 201) {
        setSuccessMessage('Vote submitted successfully!'); 
        setErrorMessage(null); 
        setSelectedOption(null); 
      } else {
        setErrorMessage('Error submitting vote. Please try again.'); 
      }
    } catch (error) {
      if (error.response && error.response.status === 401) {
        navigate('/login');
      } else {
        setErrorMessage('Error submitting vote. Please try again.');
        console.error('Error submitting vote:', error);
      }
    }
  };

  if (!poll && !errorMessage) {
    return <div>Loading poll...</div>;
  }

  return (
    <div className={`poll-container ${darkMode ? 'dark' : ''}`}> 
      {errorMessage && <div className="error-message">{errorMessage}</div>}
      {successMessage && <div className="success-message">{successMessage}</div>}
      {poll && (
        <form onSubmit={handleSubmit}>
          <h2>{poll.question}</h2>
          {poll.choices.map((choice) => (
            <div key={choice.id}>
              <input
                type="radio"
                id={`choice-${choice.id}`}
                name="poll-option"
                value={choice.id} 
                checked={selectedOption === choice.id}
                onChange={handleOptionChange}
              />
              <label htmlFor={`choice-${choice.id}`}>{choice.text}</label>
            </div>
          ))}
          <button type="submit" disabled={!selectedOption}>Submit</button>
        </form>
      )}
    </div>
  );
};

export default Poll;