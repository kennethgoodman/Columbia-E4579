import { useState , useEffect, useRef, useCallback} from "react";
import Post from "./components/Post/Post";
import "./App.scss";

export default function App() {
  const [isLoading, setLoading] = useState(true);
  const [data, setData] = useState([]);
  const [pageNum, setPageNum] = useState(1);

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
      const url = "/api/get_images";
      setLoading(true)
      const response = await fetch(
        `${url}?page=${pageNum}&limit=10`
      );
      const results = await response.json();
      setData(prevData => [...prevData, ...results])
      setLoading(false);
    }

    fetchPosts()
  },[pageNum])

  return (
    <div className="App">
        <div>
          {
            data?.map((post, index) => { 
                // we request a new set of images
                // when the second to last image is on the screen
                // kenny knows why index + 2 ???
                if (data.length === index + 2) {
                  return (
                    <div key={post.id} ref={lastElementRef} >
                      <Post content_id={post.id} post={post} />
                    </div>
                  )
                }
                return (
                  <div key={post.id}> 
                    <Post content_id={post.id} post={post} />
                  </div>
                )
              }
            )
          }
        </div>
    </div>
  );
}
