import InfiniteScroll from "react-infinite-scroller";
import { useInfiniteQuery } from "react-query";
import Post from "./components/Post/Post";
import "./App.scss";

export default function App() {
  let url = "/api/get_images";
  const fetchPosts = async ({ pageParam = 1 }) => {
    const response = await fetch(
      `${url}?page=${pageParam}&limit=10`
    );
    const results = await response.json();
    return { results, nextPage: pageParam + 1, totalPages: 100 };
  };

  const {
    data,
    isLoading,
    isError,
    hasNextPage,
    fetchNextPage
  } = useInfiniteQuery("posts", fetchPosts, {
    getNextPageParam: (lastPage, pages) => {
      if (lastPage.nextPage < lastPage.totalPages) return lastPage.nextPage;
      return undefined;
    }
  });

  return (
    <div className="App">
        {isLoading ? (
          <p>Loading...</p>
        ) : isError ? (
          <p>There was an error</p>
        ) : (
          <InfiniteScroll hasMore={hasNextPage} loadMore={fetchNextPage}>
            {data.pages.map((page) =>
              page.results.map((post) => <Post key={post.id} content_id={post.id} post={post} />)
            )}
          </InfiniteScroll>
        )}
    </div>
  );
}
