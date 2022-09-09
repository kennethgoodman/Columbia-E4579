import React from "react";
import { useState , useEffect, useRef, useCallback} from "react";
import Post from "../Post/Post";
import axios from "axios"
import "./Feed.css";
import {getRefreshTokenIfExists} from '../../utils/tokenHandler'

const Feed = (props) => {
  const [isLoading, setLoading] = useState(true);
  const [data, setData] = useState([]);
  const [pageNum, setPageNum] = useState(0);

  const observer = useRef()
  const lastElementRef = useCallback((node) => {
    if (isLoading) return
    if(observer.current) observer.current.disconnect()

    observer.current = new IntersectionObserver(entries => {
       const first = entries[0]
        if (first.isIntersecting){
          // increment the pageNumber when
          // the intersection observer comes on screen
          setPageNum(previousPageNumber => previousPageNumber + 1)
        }
    })

    if (node) {
      observer.current.observe(node)
    }
  },[isLoading])

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
      options['url'] = `${process.env.REACT_APP_API_SERVICE_URL}/content?page=${pageNum}&limit=10&seed=${props.seed}`;
      setLoading(true)
      axios(options).then((response) => {
          const results = response.data;
          setData(prevData => [...prevData, ...results]);
          setLoading(false);
      }).catch((error) => {
          console.log(error);
      });
    }

    fetchPosts()
  },[pageNum])

  return (
    <div className="Feed">
        <div>
          {
            data?.map((post, index) => {
                // we request a new set of images
                // when the second to last image is on the screen
                // kenny knows why index + 2 ???
                if (data.length === index + 2) {
                  return (
                    <div key={post.id} ref={lastElementRef}>
                      <Post
                        // eslint-disable-next-line react/jsx-handler-names
                        content_id={post.id}
                        post={post}
                      />
                    </div>
                  )
                }
                return (
                  <div key={post.id}>
                    <Post
                        // eslint-disable-next-line react/jsx-handler-names
                      content_id={post.id}
                      post={post}
                    />
                  </div>
                )
              }
            )
          }
        </div>
    </div>
  );
}

export default Feed;