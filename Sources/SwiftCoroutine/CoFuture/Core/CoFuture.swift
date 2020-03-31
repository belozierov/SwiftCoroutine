//
//  CoFuture2.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 26.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

///
///Holder for a result that will be provided later.
///
///`CoFuture` and it's subclass `CoPromise` є імплементацією Future/Promise підходу.
///Це дозволяє виконувати асинхронно роботу immediately повернувши `CoFuture`, яке can
///be observed to be notified when result will be available. For example:
///
///```
///extension URLSession {
///
///    typealias DataResponse = (data: Data, response: URLResponse)
///
///    func dataTaskFuture(for urlRequest: URLRequest) -> CoFuture<DataResponse> {
///        let promise = CoPromise<DataResponse>()
///        let task = dataTask(with: urlRequest) {
///            if let error = $2 {
///                promise.fail(error)
///            } else if let data = $0, let response = $1 {
///                promise.success((data, response))
///            } else {
///                promise.fail(URLError(.badServerResponse))
///            }
///        }
///        task.resume()
///        //cancel task if future will cancel
///        promise.whenCanceled(task.cancel)
///        return promise
///    }
///
///}
///```
///
///За допомогою `whenComplete()` ви можете додати callback або використати `await()`
///в середині коротини для отримання результату. `CoFuture` є повністю thread-safe.
///
///## Features
///
///### **Best performance**
///Основною ціллю при створенні `CoFuture` було досягнення найкращої швидкодії.
///Було витрачено багато часу і перебрано багато варіантів для цього(для того щоб знайти найкращий).
///Як результат `CoFuture` є швидшим ніж аналогічні рішення:
///
///- CoFuture - 0.083  c.
///- Combine Future - 0.234 c. **(2.8x slower)**
///- Найпопулярніша Swift Future/Promise library on GitHub - 0.521 c. **(6.3x slower)**
///
///Тести для `CoFuture` та Combine `Future` ви можете знайти в файлі `CoFuturePerformanceTests`.
///Тест проводився на MacBook Pro (13-inch, 2017, Two Thunderbolt 3 ports) у release mode.
///
///### **Cancellable**
///За допомогою `cancel()` ви можете завершити весь upstream chain of CoFutures.
///Також ви можете handle cancelling і завершити пов’язані таски.
///
///```
///let future = URLSession.shared.dataTaskFuture(for: request)
///
///future.whenCanceled {
///    //handle when canceled
///}
///
/////will also cancel URLSessionDataTask
///future.cancel()
///```
///
///### **Awaitable**
///Ви можете використовувати `await()` всередині `Coroutine` для реалізації async/await патерну для отримання
///результату. Вона дозволяє працювати з асинхронним кодом в синхронній манері без блокування потоку.
///
///```
/////execute coroutine on main thread
///CoroutineDispatcher.main.execute {
///    //extension that returns CoFuture<URLSession.DataResponse>
///    let future = URLSession.shared.dataTaskFuture(for: request)
///
///    //await result that suspends coroutine and doesn't block the thread
///    let data = try future.await().data
///
///    //set the image on main thread
///    self.imageView.image = UIImage(data: data)
///}
///```
///
///### **Combine ready**
/// `CoFuture`легко інтегрується з Combine, так за допомогою `publisher()` ви можете створити `Publisher`,
/// який transmit результат як тільки він буде готовий. Крім цього до`Publisher` був доданий extension
/// `subscribeCoFuture()`, який дає можливість subscribe `CoFuture`, який отримає лише один результат.
/// Ви можете використовувати `await()` для цього `CoFuture`, щоб отримати результат для `Publisher`
/// всередині коротини.
///
///```
///CoroutineDispatcher.main.execute {
///    //returns Publishers.MapKeyPath<URLSession.DataTaskPublisher, Data>
///    let publisher = URLSession.shared.dataTaskPublisher(for: request).map(\.data)
///    //await data without blocking the thread
///    let data = try publisher.await()
///    //do some work with data
///}
///```
///
public class CoFuture<Value> {
    
    internal let mutex: PsxLock?
    private var callbacks: ContiguousArray<Child>?
    final private(set) var _result: Optional<Result<Value, Error>>
    
    @usableFromInline internal init(mutex: PsxLock?, result: Result<Value, Error>?) {
        self.mutex = mutex
        _result = result
    }
    
    deinit {
        callbacks?.forEach { $0.callback(.failure(CoFutureError.canceled)) }
        mutex?.free()
    }
    
}

extension CoFuture {
    
    /// Initializes a future with result.
    /// - Parameter result: The result provided by this future.
    @inlinable public convenience init(result: Result<Value, Error>) {
        self.init(mutex: nil, result: result)
    }
    
    /// Initializes a future with success value.
    /// - Parameter value: The value provided by this future.
    @inlinable public convenience init(value: Value) {
        self.init(result: .success(value))
    }
    
    /// Initializes a future with error.
    /// - Parameter error: The error provided by this future.
    @inlinable public convenience init(error: Error) {
        self.init(result: .failure(error))
    }
    
    // MARK: - result
    
    /// Returns completed result or nil if this future has not completed yet.
    public var result: Result<Value, Error>? {
        mutex?.lock()
        defer { mutex?.unlock() }
        return _result
    }
    
    @usableFromInline internal func setResult(_ result: Result<Value, Error>) {
        mutex?.lock()
        if _result != nil {
            mutex?.unlock()
        } else {
            _result = result
            mutex?.unlock()
            callbacks?.forEach { $0.callback(result) }
            callbacks = nil
        }
    }
    
    // MARK: - Callback
    
    internal typealias Callback = (Result<Value, Error>) -> Void
    private struct Child { let callback: Callback }
    
    internal func append(callback: @escaping Callback) {
        if callbacks == nil {
            callbacks = [.init(callback: callback)]
        } else {
            callbacks?.append(.init(callback: callback))
        }
    }
    
    // MARK: - cancel

    /// Returns `true` when the current future is canceled.
    @inlinable public var isCanceled: Bool {
        if case .failure(let error as CoFutureError)? = result {
            return error == .canceled
        }
        return false
    }
    
    /// Cancel цей та всі пов'язані future, засетавши всім результат з CoFutureError.canceled.
    @inlinable public func cancel() {
        setResult(.failure(CoFutureError.canceled))
    }
    
}
