//
//  CoSubscription.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 15.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

#if canImport(Combine)
import Combine

@available(OSX 10.15, iOS 13.0, *)
final class CoSubscription<S: Subscriber, T>: Subscription where S.Input == T, S.Failure == Error {
    
    private let future: CoFuture<T>
    private var subscriber: S?
    
    @inlinable init(subscriber: S, future: CoFuture<T>) {
        self.future = future
        future.whenComplete { result in
            guard let subscriber = self.subscriber else { return }
            switch result {
            case .success(let result):
                _ = subscriber.receive(result)
                subscriber.receive(completion: .finished)
            case .failure(let error):
                subscriber.receive(completion: .failure(error))
            }
        }
    }
    
    @inlinable func cancel() {
        future.lock()
        subscriber = nil
        future.unlock()
    }
    
    @inlinable func request(_ demand: Subscribers.Demand) {}
    
}
#endif
