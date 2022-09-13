/* eslint-disable react/jsx-handler-names */
import React from "react";
import { useState, useEffect, useRef, useCallback } from "react";
import Post from "../Post/Post";
import axios from "axios";
import { getRefreshTokenIfExists } from "../../utils/tokenHandler";

import "./Feed.css";

const Feed = (props) => {
  const [isLoading, setLoading] = useState(true);
  const [data, setData] = useState([]);
  const [fetchParams, setFetchParams] = useState({
    page: 0,
    controller: "RANDOM",
  });

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
      ] = `${process.env.REACT_APP_API_SERVICE_URL}/content?page=${fetchParams["page"]}&limit=50&seed=${props.seed}&controller=${fetchParams["controller"]}`;
      setLoading(true);
      axios(options)
        .then((response) => {
          const results = response.data;
          setData((prevData) => [...prevData, ...results]);
          setLoading(false);
        })
        .catch((error) => {
          console.log(error);
        });
    };

    fetchPosts();
  }, [fetchParams]);

  const handleChange = (event) => {
    setData([]);
    setFetchParams({
      page: 0,
      controller: event.target.value,
    });
  };

  return (
    <div className="Feed">
      <label>
        Which Controller Do You Want To Use:
        <select value={fetchParams["controller"]} onChange={handleChange}>
          <option value="RANDOM">Random</option>
          <option value="STATIC">Static</option>
        </select>
      </label>
      {data?.map((post, index) => {
        // we request a new set of images
        // when the second to last image is on the screen
        // kenny knows why index + 2 ???
        if (data.length === index + 2) {
          return (
            <div key={post.id} ref={lastElementRef}>
              <Post content_id={post.id} post={post} />
            </div>
          );
        }
        return (
          <div key={post.id}>
            <Post content_id={post.id} post={post} />
          </div>
        );
      })}
    </div>
  );
};

export default Feed;
