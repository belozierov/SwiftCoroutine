//
//  CoFuture+Combine.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 15.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

#if canImport(Combine)
import Combine

@available(OSX 10.15, iOS 13.0, *)
extension CoFuture: Cancellable {
    
    // MARK: - publisher
    
    /// Returns a publisher that emits result of this `CoFuture`.
    public func publisher() -> AnyPublisher<Value, Error> {
        CoFuturePublisher(future: self).eraseToAnyPublisher()
    }
    
}

@available(OSX 10.15, iOS 13.0, *)
extension Publisher {
    
    /// Attaches `CoFuture` as a subscriber and returns it. `CoFuture` will receive result only once.
    public func subscribeCoFuture() -> CoFuture<Output> {
        let promise = CoPromise<Output>()
        let cancellable = mapError { $0 }.sink(receiveCompletion: { result in
            switch result {
            case .finished: promise.cancel()
            case .failure(let error): promise.fail(error)
            }
        }, receiveValue: promise.success)
        promise.whenCanceled(cancellable.cancel)
        return promise
    }
    
}
#endif
