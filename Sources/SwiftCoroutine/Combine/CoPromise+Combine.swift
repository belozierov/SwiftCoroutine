//
//  CoPromise+Combine.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 28.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Combine

@available(OSX 10.15, iOS 13.0, *)
extension CoPromise: Subject {
    
    public func send(completion: Subscribers.Completion<Error>) {
        switch completion {
        case .finished: break
        case .failure(let error): complete(with: .failure(error))
        }
    }
    
    public func send(subscription: Subscription) {
        subscription.request(.max(1))
    }
    
}
