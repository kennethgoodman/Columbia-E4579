/* eslint-disable react/jsx-handler-names */
import React from "react";
import { useState, useEffect, useRef, useCallback, useContext } from "react";
import Post from "../Post/Post";
import axios from "axios";
import { getRefreshTokenIfExists } from "../../utils/tokenHandler";
import { DarkModeContext } from '../../components/darkmode/DarkModeContext';

import "./Feed.css";

function shuffleArray(array) {
    for (var i = array.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

const Feed = (props) => {
  const { darkMode, setDarkMode } = useContext(DarkModeContext);
  const [isLoading, setLoading] = useState(true);
  const [data, setData] = useState([]);
  const [selectionValues, setSelectionValues] = useState([]);
  const [fetchParams, setFetchParams] = useState({
    page: 0,
    controller: undefined,
    starting_content_id: undefined,
  });
  const [twoTower, setTwoTower] = useState(true);
  const [collabFilter, setCollabFilter] = useState(true);
  const [yourChoice, setYourChoice] = useState(true);

  const [policyFilterOne, setPolicyFilterOne] = useState(true);
  const [policyFilterTwo, setPolicyFilterTwo] = useState(true);
  const [linearRegression, setLinearRegression] = useState(true);

  const handleCheckboxChange = (setCheckboxState, checkboxValue) => {
    setData([]);  // Clear the current data
    setFetchParams(prevState => ({
      ...prevState,
      page: 0  // Reset page to 0
    }));
    setCheckboxState(!checkboxValue);  // Toggle the checkbox value
  };

  useEffect(() => {
    const options = {
      method: "get",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${getRefreshTokenIfExists()}`,
      },
      url: `${process.env.REACT_APP_API_SERVICE_URL}/content/listcontrollers`
    };

    axios(options)
      .then((response) => {
        const results = response.data;
        let selection_values = [];
        results.forEach((item) => {
          selection_values.push(
            {'key': item["controller"].toUpperCase(), 'value': item["controller"]}
          )
        });
        setSelectionValues(selection_values);
        setFetchParams(prevState => ({
          ...prevState,
          controller: selection_values[0]['value'],
        }));
      })
      .catch((error) => {
        alert(`error with listcontrollers ${error}`);
      });
  }, []); // The empty dependency array ensures this runs only once when the component mounts

  const handleSeeMore = (content_id) => {
    if (content_id === fetchParams["starting_content_id"]) {
      return;
    }
    setData([]);
    setFetchParams((previousFetchParams) => {
      return {
        page: 0,
        controller: previousFetchParams["controller"],
        starting_content_id: content_id,
      };
    });
  };

  const observer = useRef();
  const lastElementRef = useCallback(
    (node) => {
      if (isLoading) return;
      if (observer.current) observer.current.disconnect();

      observer.current = new IntersectionObserver((entries) => {
        const first = entries[0];
        if (first.isIntersecting) {
          // increment the pageNumber when
          // the intersection observer comes on screen
          setFetchParams((previousFetchParams) => {
            return {
              page: previousFetchParams["page"] + 1,
              controller: previousFetchParams["controller"],
              starting_content_id: previousFetchParams["starting_content_id"],
            };
          });
        }
      });

      if (node) {
        observer.current.observe(node);
      }
    },
    [isLoading],
  );

  // every time the pageNum changes
  // we request a new set of images
  useEffect(() => {
    if(fetchParams["controller"] === undefined) {
      return
    }
    const fetchPosts = async () => {
      const options = {
        method: "get",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${getRefreshTokenIfExists()}`,
        },
      };
      options[
        "url"
      ] = `${process.env.REACT_APP_API_SERVICE_URL}/content?page=${fetchParams["page"]}&limit=50&seed=${props.seed}&controller=${fetchParams["controller"]}&content_id=${fetchParams["starting_content_id"]}&twoTower=${twoTower}&collabFilter=${collabFilter}&yourChoice=${yourChoice}&policyFilterOne=${policyFilterOne}&policyFilterTwo=${policyFilterOne}&linearRegression=${linearRegression}`;
      setLoading(true);
      axios(options)
        .then((response) => {
          const results = response.data;
          setData((prevData) => [...prevData, ...results]);
          setLoading(false);
        })
        .catch((error) => {
          alert(error);
          setLoading(false);
        });
    };

    fetchPosts();
  }, [fetchParams, twoTower, collabFilter, yourChoice]);

  const handleChange = (event) => {
    setData([]);
    setFetchParams((previousFetchParams) => {
      return {
        page: 0,
        controller: event.target.value,
        starting_content_id: previousFetchParams["starting_content_id"],
      };
    });
  };


  const getTimeEngaged = async () => {
    const options = {
      method: "get",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${getRefreshTokenIfExists()}`,
      },
    };
    options[
      "url"
    ] = `${process.env.REACT_APP_API_SERVICE_URL}/engagement/time_engaged/${fetchParams["controller"]}`;
    setLoading(true);
    axios(options)
      .then((response) => {
        const results = response.data;
        var minutes = Math.floor(results/60000);
        var seconds = Math.floor((results % 60000)/1000);
        setButtonText(minutes+"m"+seconds+"s");
        setLoading(false);
      })
      .catch((error) => {
        console.log("error in getTimeEngaged"+error);
        setButtonText("please log in");
        setLoading(false);
      });
  };


  const [buttonText, setButtonText] = useState("time engaged")

  return (
    <div className={darkMode ? 'Feed dark' : 'Feed'}>
      <button
        className="primary"
        style={{width: "60px", height: "40px", margin: 0, top: 'auto', right: 20, bottom: 20, left: 'auto', position: 'fixed', color:'#7e95be'}}
        onMouseEnter={() => getTimeEngaged()}
        onMouseLeave={() => setButtonText("time engaged")}
      >
          {buttonText}
      </button>
      <div className="checkbox-group">
        <p>Candidate Generators:</p>
        <label>
          <input type="checkbox" checked={twoTower} onChange={() => handleCheckboxChange(setTwoTower, twoTower)} />
          Two Tower
        </label>
        <label>
          <input type="checkbox" checked={collabFilter} onChange={() => handleCheckboxChange(setCollabFilter, collabFilter)} />
          Collaborative Filter
        </label>
        <label>
          <input type="checkbox" checked={yourChoice} onChange={() => handleCheckboxChange(setYourChoice, yourChoice)} />
          Your Choice
        </label>
      </div>
      <div className="checkbox-group">
        <p>Filters:</p>
        <label>
          <input type="checkbox" checked={policyFilterOne} onChange={() => handleCheckboxChange(setPolicyFilterOne, policyFilterOne)} />
          Policy Filter One
        </label>
        <label>
          <input type="checkbox" checked={policyFilterTwo} onChange={() => handleCheckboxChange(setPolicyFilterTwo, policyFilterTwo)} />
          Policy Filter Two
        </label>
        <label>
          <input type="checkbox" checked={linearRegression} onChange={() => handleCheckboxChange(setLinearRegression, linearRegression)} />
          Linear Regression
        </label>
      </div>
      <label className="switch">
        <input type="checkbox" checked={darkMode} onChange={() => setDarkMode(!darkMode)} />
        <span className="slider round" />
      </label>
      <label>
        Which Controller Do You Want To Use:
        <select value={fetchParams["controller"]} onChange={handleChange}>
          {
            selectionValues.map(el => {
              return <option value={el['value']} key={el['key']}>{el['key']}</option>
            })
          }
        </select>
      </label>
      {fetchParams["starting_content_id"] !== undefined && (
        <button onClick={() => handleSeeMore(undefined)}>
          Seeing more for {fetchParams["starting_content_id"]}, Click Here To Go
          Back
        </button>
      )}
      {data?.length === 0 && (
        <div className={`empty-state ${darkMode ? 'dark' : ''}`}>
          <p>No items found. Needs To Be Implemented By The Team.</p>
        </div>
      )}
      {data?.map((post, index) => {
        // we request a new set of images when the second to last image is on the screen
        // TODO, maybe switch to 80%
        if (data.length === index + 2) {
          return (
            <div key={post.id} ref={lastElementRef}>
              <Post
                content_id={post.id}
                post={post}
                handleSeeMore={handleSeeMore}
                controller={post['controller'] ? post['controller'] : fetchParams["controller"]}
              />
            </div>
          );
        }
        return (
          <div key={post.id}>
            <Post
              content_id={post.id}
              post={post}
              handleSeeMore={handleSeeMore}
              controller={fetchParams["controller"]}
            />
          </div>
        );
      })}
    </div>
  );
};

export default Feed;
