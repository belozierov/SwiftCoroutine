//
//  CoFuturePublisher.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 15.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

#if canImport(Combine)
import Combine

@available(OSX 10.15, iOS 13.0, *)
internal final class CoFuturePublisher<Output> {
    
    internal typealias Failure = Error
    
    internal let future: CoFuture<Output>
    
    @inlinable internal init(future: CoFuture<Output>) {
        self.future = future
    }
    
}

@available(OSX 10.15, iOS 13.0, *)
extension CoFuturePublisher: Publisher {
    
    @inlinable internal func receive<S: Subscriber>(subscriber: S) where Failure == S.Failure, Output == S.Input {
        let subscription = CoFutureSubscription(subscriber: subscriber, future: future)
        subscriber.receive(subscription: subscription)
    }
    
}

//@available(OSX 10.15, iOS 13.0, *)
//final class CoFutureSubject<T>: CoFuturePublisher<T, CoPromise<T>> {}
//
//@available(OSX 10.15, iOS 13.0, *)
//extension CoFutureSubject: Subject {
//
//    @inlinable func send(_ value: Output) {
//        future.success(value)
//    }
//
//    @inlinable func send(completion: Subscribers.Completion<Failure>) {
//        switch completion {
//        case .finished: future.cancel()
//        case .failure(let error): future.complete(with: .failure(error))
//        }
//    }
//
//    @inlinable func send(subscription: Subscription) {
//        subscription.request(.max(1))
//    }
//
//}
#endif
