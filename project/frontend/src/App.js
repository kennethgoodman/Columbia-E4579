import "./App.css";
import useFetch from "./utils/useFetch";

function App() {
  const { data: joke, loading, error, refetch } = useFetch(
    "/api/joke"
  );

  if (loading) return <h1> LOADING...</h1>;

  if (error) console.log(error);

  return (
    <div className="App">
      <h1>
        {joke?.setup} : {joke?.delivery}
      </h1>

      <button onClick={refetch}> Refetch</button>
    </div>
  );
}

export default App;