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
    
    public func receive<S: Subscriber>(subscriber: S) where Output == S.Input, Failure == S.Failure {
        let subscription = CoSubscription { [weak self] in
            self?.completions[$0] = nil
        }
        subscriber.receive(subscription: subscription)
        completions[subscription] = subscriber.finish
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
