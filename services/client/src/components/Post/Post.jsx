import React, { useEffect, useRef, useState, useMemo } from "react";
import LikeButton from "../Likes/LikeButton";
import DislikeButton from "../Likes/DislikeButton";
import axios from "axios";
import { getRefreshTokenIfExists } from "../../utils/tokenHandler";

import "./Post.css";

const get_options = (uri, content_id) => {
  let api_uri = `${process.env.REACT_APP_API_SERVICE_URL}/engagement/${uri}/${content_id}`;
  return {
    url: api_uri,
    method: "post",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${getRefreshTokenIfExists()}`,
    },
  };
};

const Post = (props) => {
  const [likeIsClicked, setLikeIsClicked] = useState(props.post.user_likes);
  const [dislikeIsClicked, setDislikeIsClicked] = useState(
    props.post.user_dislikes,
  );
  const [totalLikes, setTotalLikes] = useState(props.post.total_likes);
  const [totalDislikes, setTotalDislikes] = useState(props.post.total_dislikes);
  const [isAuthenticated, _] = useState(getRefreshTokenIfExists() !== null);

  const image_ref = useRef(null);
  useIntersectionObserver(props.content_id, image_ref, {});

  const like = (callback) => {
    axios(get_options("like", props.content_id))
      .then(callback)
      .catch(function (error) {
        console.log(error);
      });
  };
  const unlike = (callback) => {
    axios(get_options("unlike", props.content_id))
      .then(callback)
      .catch(function (error) {
        console.log(error);
      });
  };
  const dislike = (callback) => {
    axios(get_options("dislike", props.content_id))
      .then(callback)
      .catch(function (error) {
        console.log(error);
      });
  };
  const undislike = (callback) => {
    axios(get_options("undislike", props.content_id))
      .then(callback)
      .catch(function (error) {
        console.log(error);
      });
  };

  const handleLikes = () => {
    if (likeIsClicked) {
      unlike((_) => {
        setLikeIsClicked(false); // unclick it
        setTotalLikes(totalLikes - 1);
      });
    } else {
      like((_) => {
        setLikeIsClicked(true); // click it
        setTotalLikes(totalLikes + 1);
      });
      if (dislikeIsClicked) {
        setDislikeIsClicked(false);
        setTotalDislikes(totalDislikes - 1);
      }
    }
  };

  const handleDislikes = () => {
    console.log(
      `handing dislike for ${props.content_id}, ${likeIsClicked}, ${dislikeIsClicked}`,
    );
    if (dislikeIsClicked) {
      undislike((_) => {
        setDislikeIsClicked(false); // unclick it
        setTotalDislikes(totalDislikes - 1);
      });
    } else {
      dislike((_) => {
        setDislikeIsClicked(true); // click it
        setTotalDislikes(totalDislikes + 1);
      });
      if (likeIsClicked) {
        setLikeIsClicked(false);
        setTotalLikes(totalLikes - 1);
      }
    }
  };

  return (
    <div className="postContainer">
      <h4 className="postAuthor">{props.post.author}</h4>
      <button onClick={() => props.handleSeeMore(props.content_id)}>
        See More Like This
      </button>
      <img
        ref={image_ref}
        src={props.post.download_url}
        alt={props.post.text}
        onDoubleClick={handleLikes}
      />
      <p className="postBody">{props.post.text}</p>
      {isAuthenticated && (
        <div className="likesContainer">
          <LikeButton
            content_id={props.content_id}
            total_likes={totalLikes}
            user_likes={likeIsClicked}
            handleLikes={handleLikes}
          />
          <DislikeButton
            content_id={props.content_id}
            total_dislikes={totalDislikes}
            user_dislikes={dislikeIsClicked}
            handleDislikes={handleDislikes}
          />
        </div>
      )}
    </div>
  );
};

function useIntersectionObserver(
  content_id,
  elementRef,
  {
    threshold = 0.9,
    root = null,
    rootMargin = "0%",
    freezeOnceVisible = false,
  },
) {
  const [entry, setEntry] = useState();
  const [startIntersecting, setStartIntersecting] = useState(-1);

  const frozen = entry?.isIntersecting && freezeOnceVisible;

  const handle_elapsed = (elapsed_time, content_id) => {
    if (elapsed_time <= 1000) {
      return;
    }
    console.log("in handle elapsed", elapsed_time, content_id);
    let api_uri = "/api/engagement/elapsed_time";
    axios({
      ...get_options("elapsed_time", content_id),
      data: JSON.stringify({ elapsed_time: elapsed_time }),
    }).catch(function (error) {
      console.log(error);
    });
  };

  const updateEntry = ([entry]) => {
    setEntry(entry);
    if (entry.isIntersecting) {
      if (startIntersecting !== -1) {
        return;
      }
      setStartIntersecting(Date.now());
      // handleLoad();
    } else if (startIntersecting === -1) {
      return;
      // first time loading, but not in view
    } else {
      // not intersecting, and not loaded for first time
      handle_elapsed(Date.now() - startIntersecting, content_id);
      setStartIntersecting(-1);
    }
  };

  useEffect(() => {
    // https://usehooks-ts.com/react-hook/use-intersection-observer
    const node = elementRef?.current; // DOM Ref
    const hasIOSupport = !!window.IntersectionObserver;

    if (!hasIOSupport || frozen || !node) return;

    const observerParams = { threshold, root, rootMargin };
    const observer = new IntersectionObserver(updateEntry, observerParams);

    observer.observe(node);

    return () => observer.disconnect();
  }, [
    elementRef,
    JSON.stringify(threshold),
    root,
    rootMargin,
    frozen,
    startIntersecting,
    content_id,
  ]);

  return entry;
}

// const defaultRequestOptions = {
//   headers: {
//     "Content-Type": "application/json",
//     Accept: "application/json",
//   },
// };
// const postRequestOptions = {
//   ...defaultRequestOptions,
//   method: "POST",
// };
// const handleLoad = () => {
//   if (loaded) return; // only load once for session
//   setLoaded(true);
//   let api_uri = "/api/engagement/loaded_content";
//   // fetch(
//   //   `${api_uri}?content_id=${content_id}`,
//   //   postRequestOptions,
//   // ).then((response) => response.json());
// };
// const handle_elapsed = (elapsed_time) => {
//             if (elapsed_time <= 200) {
//               return;
//             }
//             console.log("in handle elapsed", elapsed_time);
//             let api_uri = "/api/engagement/elapsed_time";
//             // fetch(`${api_uri}?content_id=${content_id}`, {
//             //   ...postRequestOptions,
//             //   body: JSON.stringify({ elapsed_time: elapsed_time }),
//             // }).then((response) => response.json());
//           };
//
// function useIsInViewport(ref, content_id) {
//   const [renderColorSets, setRenderColorSets] = useState(5);
//   useEffect(() => {
//     const observer = new IntersectionObserver((entries) => {
//       if (entries[0].isIntersecting) {
//         setRenderColorSets((oldRenderColorSets) => oldRenderColorSets + 1);
//         console.log(content_id, renderColorSets);
//       }
//     });
//
//     observer.observe(ref.current);
//   }, []);
// }

export default Post;

// TODO: on leaving page (or send updates every 100ms?)
// TODO: update backend
