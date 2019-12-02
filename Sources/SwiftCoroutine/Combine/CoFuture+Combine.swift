//
//  CoFuture+Combine.swift
//  SwiftCoroutine iOS
//
//  Created by Alex Belozierov on 28.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Combine

@available(OSX 10.15, iOS 13.0, *)
extension CoFuture: Publisher, Cancellable {
    
    public typealias Failure = Error
    
    open func receive<S: Subscriber>(subscriber: S) where Output == S.Input, Failure == S.Failure {
        mutex.lock()
        defer { mutex.unlock() }
        if let result = _result { return subscriber.finish(with: result) }
        let subscription = CoSubscription { [weak self] in self?.removeSubscription(for: $0) }
        subscriptions[subscription] = subscriber.finish
        subscriber.receive(subscription: subscription)
    }
    
    private func removeSubscription(for key: CoSubscription) {
        subscriptions.removeValue(forKey: key)
        if subscriptions.isEmpty { cancel() }
    }
    
}

@available(OSX 10.15, iOS 13.0, *)
extension Subscriber {
    
    fileprivate func finish(with result: Result<Input, Failure>) {
        switch result {
        case .success(let input):
            _ = receive(input)
            receive(completion: .finished)
        case .failure(let error):
            receive(completion: .failure(error))
        }
    }
    
}
