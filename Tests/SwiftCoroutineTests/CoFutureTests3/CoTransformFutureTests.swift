////
////  CoTransformFutureTests.swift
////  SwiftCoroutine
////
////  Created by Alex Belozierov on 10.01.2020.
////  Copyright © 2020 Alex Belozierov. All rights reserved.
////
//
//import XCTest
//@testable import SwiftCoroutine
//
//class CoTransformFutureTests: CoFutureTests {
//    
//    override var future: CoTransformFuture<Int, Bool> {
//        futures.transform
//    }
//    
//    private var futures: (parent: CoFuture<Int>, transform: CoTransformFuture<Int, Bool>) {
//        let parent = CoFuture<Int>()
//        return (parent, transform(parent: parent))
//    }
//    
//    private func transform(parent: CoFuture<Int>) -> CoTransformFuture<Int, Bool> {
//        CoTransformFuture(parent: parent) { try $0.get() != 0 }
//    }
//    
//    func testInit() {
//        let (parent, transform) = futures
//        XCTAssertEqual(transform.subscriptions?.count, 0)
//        XCTAssertEqual(parent.subscriptions?.count, 1)
//        XCTAssertEqual(parent.subscriptions?.first?.key, transform)
//        XCTAssertEqual(parent.mutex, transform.mutex)
//        XCTAssertFalse(parent.$resultStorage.projectedValue
//            === transform.$resultStorage.projectedValue)
//        XCTAssertNil(transform.result)
//    }
//    
//    func testInitWithResult() {
//        let parent = CoFuture<Int>()
//        parent.complete(with: .success(1))
//        let transform = self.transform(parent: parent)
//        XCTAssertEqual(parent.subscriptions?.count, 0)
//        XCTAssertEqual(transform.subscriptions?.count, 0)
//        XCTAssertEqual(transform.result, true)
//    }
//    
//    func testCancelUnsubscribe() {
//        let (parent, transform) = futures
//        transform.cancel()
//        XCTAssertTrue(parent.subscriptions?.isEmpty == true)
//    }
//    
//}
