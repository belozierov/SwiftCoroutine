<!--
  Title: SwiftCoroutine
  Description: Swift coroutines for iOS and macOS.
  Author: belozierov
  Keywords: swift, coroutines, coroutine, async/await
  -->
  
![Swift Coroutine](../master/Sources/logo.png)

##
Many languages, such as Kotlin, JavaScript, Go, Rust, C++, and others, already have [coroutines](https://en.wikipedia.org/wiki/Coroutine) support that makes the use of asynchronous code easier. Unfortunately, Apple is still behind on this feature. But this can be improved by a framework without the need to change the language.

This is the first implementation of [coroutines](https://en.wikipedia.org/wiki/Coroutine) for Swift with macOS and iOS support. They make the [async/await](https://en.wikipedia.org/wiki/Async/await) pattern implementation possible. In addition, the framework includes [futures and promises](https://en.wikipedia.org/wiki/Futures_and_promises) for more flexibility and ease of use.

### Usage

```swift
//perform coroutine on the main thread
//submit() returns CoFuture<Void>, thanks to which we can handle errors
CoroutineDispatcher.main.submit {

    //extension that returns CoFuture<(data: Data, response: URLResponse)>
    let dataFuture = URLSession.shared.dataTaskFuture(for: imageURL)
    
    //await result that suspends coroutine and doesn't block the thread
    let data = try dataFuture.await().data

    //create UIImage from data or throw the error
    guard let image = UIImage(data: data) else { throw URLError(.cannotParseResponse) }
    
    //perform task on global queue and await the result without blocking the thread
    let thumbnail = try TaskScheduler.global.await { image.makeThumbnail() }

    //coroutine is performed on the main thread, that's why we can set the image in UIImageView
    self.imageView.image = thumbnail
    
}.whenFailure { error in
    //here we can handle errors
    self.imageView.image = UIImage(name: "thumbnail_placeholder")
}
```

### Requirements

- iOS 11.0+ / macOS 10.13+
- Xcode 10.2+
- Swift 5+

### Installation

`SwiftCoroutine` is available through the [Swift Package Manager](https://swift.org/package-manager) for macOS and iOS.

## Working with SwiftCoroutine

### Async/await

Asynchronous programming is usually associated with callbacks. It is quite convenient until there are too many of them and they start nesting. Then it's called **callback hell**.

The **async/await** pattern is an alternative. It is already well-established in other programming languages and is an evolution in asynchronous programming. The implementation of this pattern is possible thanks to coroutines. 

#### Key benefits
- **Suspend instead of block**. The main advantage of coroutines is the ability to suspend their execution at some point without blocking a thread and resuming later on.
- **Fast context switching**. Switching between coroutines is much faster than switching between threads as it does not require the involvement of operating system.
- **Asynchronous code in synchronous manner**. The use of coroutines allows an asynchronous, non-blocking function to be structured in a manner similar to an ordinary synchronous function. And even though coroutines can run in multiple threads, your code will still look consistent and therefore easy to understand.

You can execute tasks inside coroutines on `CoroutineDispatcher` just like you would do it on `DispatchQueue`. While `Coroutine.await()` allows you to wrap asynchronous functions to deal with them as synchronous. 

#### Main features
- **Any scheduler**. You can use any scheduler to execute coroutines wrapping it in TashScheduler, including standard `DispatchQueue` or even `NSManagedObjectContext` and `MultiThreadedEventLoopGroup`.
- **Memory efficiency**. Contains a mechanism that allows to reuse stacks and, if necessary, effectively store their contents.
- **Await instead of resume/suspend**. For convenience and safety, coroutines' resume/suspend has been replaced by await, which suspends it and resumes on callback.
- **Lock-free await**. Await is implemented using atomic variables. This makes it especially fast in cases where the result is already available.
- **Create your own API**. Gives you a very flexible tool to create your own add-ons or integrate with existing solutions.

```swift
func makeSomeFuture() -> CoFuture<Int> {
    let promise = CoPromise<Int>()
    someAsyncFuncWithCompletion { int in
        promise.send(int)
    }
    return promise
}

let future = makeSomeFuture().transformOutput { $0.description } 
future.onResult(on: .global) { result in
    //do some work with result of type Result<String, Error>
}
```

### Futures and Promises

The futures and promises approach takes the usage of asynchronous code to the next level. It is a convenient mechanism to synchronise asynchronous code and has become a part of the async/await pattern. If coroutines are a skeleton, then futures and promises are its muscles.

Futures and promises are represented by the corresponding `CoFuture` class and its `CoPromise` subclass. `CoFuture` is a holder for a result that will be provided later.

#### Main features
- **Best performance**. It is much faster than most of other futures and promises implementations.
- **Build chains**. With `flatMap()` and `map()`, you can create data dependencies via `CoFuture` chains.
- **Cancellable**. You can cancel the whole chain as well as handle it and complete the related actions.
- **Awaitable**. You can await the result inside the coroutine.
- **Combine-ready**. You can create `Publisher` from `CoFuture`, and vice versa make `CoFuture` a subscriber.

Unlike `Coroutine.await()`, with `CoFuture.await()`, you can start multiple tasks in parallel and synchronise them later.

```swift
let future1: CoFuture<Int> = async {
    sleep(2) //some work
    return 5
}

let future2: CoFuture<Int> = async {
    sleep(3) //some work
    return 6
}

coroutine(on: .main) {
    let sum = try future1.await() + future2.await() //will await for 3 sec., doesn't block the thread
    self.label.text = "Sum is \(sum)"
}
```
